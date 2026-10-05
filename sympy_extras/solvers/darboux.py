"""Darboux polynomials and the Prelle–Singer procedure: elementary and
Liouvillian first integrals of first order equations ``y' = P(x, y)/Q(x,
y)`` with polynomial ``P`` and ``Q``.

The equation is the vector field ``D = Q d/dx + P d/dy`` (a first
integral ``I(x, y)`` is a function with ``D I = 0``). A *Darboux
polynomial* of ``D`` is a polynomial ``f`` with ``D f = g f`` for a
polynomial ``g``, its *cofactor*, of degree at most ``d - 1`` where ``d``
is the degree of the field (the larger of the degrees of ``P`` and
``Q``); its zero set is a union of solution curves. An *exponential
factor* is a function ``exp(A/B)`` with ``D(A/B) = L`` a polynomial of
degree at most ``d - 1`` (the denominator ``B`` is a product of Darboux
polynomials). The *Darboux integrating factors* are the products ``R =
prod f_i**c_i prod exp(e_j A_j/B_j)``: ``R (Q dy - P dx)`` is exact
exactly when ``sum c_i g_i + sum e_j L_j = -div D = -(Q_x + P_y)``, a
linear system for the exponents, and ``sum c_i g_i + sum e_j L_j = 0``
makes ``sum c_i log f_i + sum e_j A_j/B_j`` a first integral at once.

Prelle and Singer [PrelleSinger]_ proved that an equation with an
elementary first integral has an integrating factor ``R`` with ``R**n``
rational for some integer ``n``, hence a Darboux integrating factor with
rational exponents and no exponential factors, and Singer [Singer]_ that
a Liouvillian first integral comes with an integrating factor of the
form ``exp(int U dx + V dy)`` with ``U``, ``V`` rational, which is a
Darboux integrating factor with exponential factors (Christopher
[Christopher]_). The procedure is thus complete up to the degree bound
on the Darboux polynomials, for which no a priori bound is known.

The Darboux polynomials of degree ``N`` are found by undetermined
coefficients with the reduction of Man [Man]_: the homogeneous part of
highest degree ``f_N`` of ``f`` satisfies ``D_d f_N = g_{d-1} f_N`` for the
highest degree parts ``D_d`` of the field and ``g_{d-1}`` of the
cofactor, so every linear factor of ``f_N`` divides ``H = x P_d - y Q_d``
(the equation of the line at infinity's tangencies): ``f_N`` is a product
of irreducible factors of ``H``, enumerated, and ``g_{d-1}`` is
``D_d f_N / f_N``. When ``H`` vanishes identically (the line at infinity
is dicritical, ``D_d = k (x d/dx + y d/dy)``) ``f_N`` is any form of
degree ``N`` and ``g_{d-1} = N k``, and the charts of the projective
space of forms are enumerated instead. The remaining coefficients
satisfy a bilinear system: its linear equations are solved and
substituted repeatedly, and what is left (products of the free
parameters of earlier degrees) is linear in the coefficients of ``f``
with coefficients polynomial in the few cofactor coefficients left,
and goes through a Gaussian elimination with a case split at every
pivot which may vanish, whose leaves are zero-dimensional systems in
the cofactor coefficients (the cofactors of bounded degree are finitely
many), solved for their points in the coefficient field by factoring
the equations and taking lex Gröbner bases of the cases; at each
cofactor the polynomials form an affine space, of which a few members
are factored. Exponential factors with a given denominator are the
nullspace of a linear map. Everything is computed over the field of the coefficients
(the rationals, an algebraic extension, or rational functions of
parameters); Darboux polynomials irreducible over that field but not
over its closure are found as the products of their conjugates.

References
==========

.. [PrelleSinger] M. J. Prelle and M. F. Singer, Elementary first
   integrals of differential equations, Trans. Amer. Math. Soc. 279
   (1983), 215–229.
.. [Singer] M. F. Singer, Liouvillian first integrals of differential
   equations, Trans. Amer. Math. Soc. 333 (1992), 673–688.
.. [Man] Y.-K. Man, Computing closed form solutions of first order ODEs
   using the Prelle–Singer procedure, J. Symbolic Computation 16 (1993),
   423–443.
.. [ManMacCallum] Y.-K. Man and M. A. H. MacCallum, A rational approach
   to the Prelle–Singer algorithm, J. Symbolic Computation 24 (1997),
   31–43.
.. [Christopher] C. Christopher, Liouvillian first integrals of second
   order polynomial differential equations, Electron. J. Differential
   Equations 1999, no. 49, 1–7.
.. [Darboux] G. Darboux, Mémoire sur les équations différentielles
   algébriques du premier ordre et du premier degré, Bull. Sci. Math. 2
   (1878), 60–96, 123–144, 151–200.
"""
from __future__ import annotations

from itertools import combinations_with_replacement
from math import lcm
from typing import Optional, Sequence

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative, count_ops
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp, log
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed, GeneratorsNeeded, PolynomialError
from sympy.polys.polytools import Poly, cancel, parallel_poly_from_expr
from sympy.polys.rings import PolyElement, PolyRing
from sympy.simplify.radsimp import fraction
from sympy.simplify.simplify import simplify

from sympy_extras._timeout import attempt, remaining_time
from sympy_extras._typing import DomainElement, ExprLike, as_expr
from sympy_extras.settings import settings

__all__ = ['DarbouxPolynomial', 'ExponentialFactor', 'DarbouxIntegral', 'vector_field',
           'darboux_polynomials', 'exponential_factors', 'darboux_integrating_factor',
           'darboux_first_integral', 'prelle_singer']

#: an exponent vector of a monomial ``x**i y**j``
_Monomial = tuple[int, int]
#: a solution of a linear system: a particular solution, a basis of the
#: homogeneous solutions (as vectors of domain elements) and the free
#: columns, one per basis vector, in which that vector has a one
_Affine = tuple[list[DomainElement], list[list[DomainElement]], list[int]]


class DarbouxPolynomial:
    """A Darboux polynomial ``f`` of the field ``D = Q d/dx + P d/dy``,
    irreducible over the coefficient field, and its cofactor ``g``,
    ``D f = g f``.

    Attributes
    ==========

    polynomial : Expr
        ``f``, monic in the lex order of ``x, y``.
    cofactor : Expr
        ``g``, of degree at most ``d - 1``.
    """

    def __init__(self, polynomial: Expr, cofactor: Expr) -> None:
        self.polynomial = polynomial
        self.cofactor = cofactor

    def __repr__(self) -> str:
        return "DarbouxPolynomial(%s, %s)" % (self.polynomial, self.cofactor)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DarbouxPolynomial):
            return NotImplemented
        return bool(self.polynomial == other.polynomial and self.cofactor == other.cofactor)

    def __hash__(self) -> int:
        return hash((self.polynomial, self.cofactor))


class ExponentialFactor:
    """An exponential factor ``exp(A/B)`` of the field, ``D(A/B) = L`` a
    polynomial of degree at most ``d - 1``, its cofactor.

    Attributes
    ==========

    numerator, denominator : Expr
        ``A`` and ``B``; ``B`` is ``1`` or a product of Darboux
        polynomials.
    cofactor : Expr
        ``L``.
    """

    def __init__(self, numerator: Expr, denominator: Expr, cofactor: Expr) -> None:
        self.numerator = numerator
        self.denominator = denominator
        self.cofactor = cofactor

    @property
    def argument(self) -> Expr:
        """``A/B``."""
        return as_expr(self.numerator/self.denominator)

    @property
    def factor(self) -> Expr:
        """``exp(A/B)``."""
        return as_expr(exp(self.argument))

    def __repr__(self) -> str:
        return "ExponentialFactor(%s, %s, %s)" % (self.numerator, self.denominator, self.cofactor)


class DarbouxIntegral:
    """The result of the Prelle–Singer procedure on a field: the Darboux
    integrating factor and the first integral, when found.

    Attributes
    ==========

    polynomials : list[DarbouxPolynomial]
        The Darboux polynomials found within the degree bound.
    exponential_factors : list[ExponentialFactor]
        The exponential factors found.
    integrating_factor : Expr or None
        ``R`` with ``R (Q dy - P dx)`` exact.
    first_integral : Expr or None
        ``I(x, y)`` with ``D I = 0``, not constant.
    """

    def __init__(self, polynomials: list[DarbouxPolynomial], factors: list[ExponentialFactor],
                 integrating_factor: Optional[Expr], first_integral: Optional[Expr]) -> None:
        self.polynomials = polynomials
        self.exponential_factors = factors
        self.integrating_factor = integrating_factor
        self.first_integral = first_integral

    def __repr__(self) -> str:
        return "DarbouxIntegral(integrating_factor=%s, first_integral=%s)" % (
            self.integrating_factor, self.first_integral)


class _Field:
    """The vector field ``Q d/dx + P d/dy`` with its polynomials over the
    coefficient field ``K``."""

    def __init__(self, P: Poly, Q: Poly, x: Symbol, y: Symbol) -> None:
        self.x = x
        self.y = y
        self.K: Domain = P.domain
        self.P = P
        self.Q = Q
        self.degree: int = max(_total_degree(P), _total_degree(Q))

    def apply(self, f: Poly) -> Poly:
        """``D f = Q f_x + P f_y``."""
        return _poly(self.Q*f.diff(self.x) + self.P*f.diff(self.y), self)

    def divergence(self) -> Poly:
        """``Q_x + P_y``."""
        return _poly(self.Q.diff(self.x) + self.P.diff(self.y), self)

    def cofactor_of(self, f: Poly) -> Optional[Poly]:
        """``D f / f`` when the division is exact."""
        quotient, remainder = self.apply(f).div(f)
        if not remainder.is_zero:
            return None
        return _poly(quotient, self)

    def homogeneous_part(self, f: Poly, n: int) -> Poly:
        """The terms of total degree ``n`` of ``f``."""
        terms = {monomial: coefficient for monomial, coefficient in f.terms() if sum(monomial) == n}
        return Poly.from_dict(terms, self.x, self.y, domain=self.K)

    def zero(self) -> Poly:
        return Poly(0, self.x, self.y, domain=self.K)

    def one(self) -> Poly:
        return Poly(1, self.x, self.y, domain=self.K)


def _total_degree(f: Poly) -> int:
    if f.is_zero:
        return -1
    return max(sum(monomial) for monomial, _ in f.terms())


def _poly(e: object, field: _Field) -> Poly:
    """``e`` (a polynomial or an expression) as a polynomial in ``x, y``
    over the field's domain."""
    if isinstance(e, Poly):
        return e.set_domain(field.K) if e.domain != field.K else e
    return Poly(e, field.x, field.y, domain=field.K)


def _monomials(n: int) -> list[_Monomial]:
    """The exponent vectors of the monomials of total degree at most
    ``n`` (none for a negative ``n``)."""
    return [(i, k - i) for k in range(n + 1) for i in range(k, -1, -1)]


def _coefficient_domain(P: Expr, Q: Expr, x: Symbol, y: Symbol,
                        extension: Optional[Expr]) -> tuple[Poly, Poly]:
    """``P`` and ``Q`` as polynomials in ``x, y`` over a common field,
    extended by ``extension`` when given (``I``, ``sqrt(2)``)."""
    try:
        if extension is None:
            (p, q), _ = parallel_poly_from_expr([P, Q], x, y)
            if p.domain.is_EX:
                # algebraic numbers among the coefficients
                (p, q), _ = parallel_poly_from_expr([P, Q], x, y, extension=True)
        else:
            (p, q), _ = parallel_poly_from_expr([P, Q], x, y, extension=extension)
    except (PolynomialError, GeneratorsNeeded, NotImplementedError) as error:
        raise ValueError("polynomials in %s and %s over a field are expected" % (x, y)) from error
    domain: Domain = p.domain
    if domain.is_EX:
        raise ValueError("polynomials in %s and %s over a field are expected" % (x, y))
    if not domain.is_Field:
        domain = domain.get_field()
    return p.set_domain(domain), q.set_domain(domain)


def vector_field(equation: Basic, f: AppliedUndef) -> Optional[tuple[Expr, Expr, Symbol, Symbol]]:
    """``(P, Q, x, y)`` of the first order equation ``y' = P(x, y)/Q(x, y)``
    with ``P`` and ``Q`` coprime polynomials in ``x`` and ``y`` (``y`` a
    fresh symbol for ``f``), or ``None`` when the equation is not of this
    form.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.darboux import vector_field
    >>> y = Function('y')(x)
    >>> vector_field((x - y)*y.diff(x) - x - y, y)
    (x + y, x - y, x, y)
    """
    from .first_order import _derivative_polynomial, _explicit_rhs
    F, x, u, p = _derivative_polynomial(equation, f)
    rhs = _explicit_rhs(F, p)
    if rhs is None:
        return None
    numerator, denominator = fraction(cancel(rhs))
    P, Q = as_expr(numerator), as_expr(denominator)
    if not (P.is_polynomial(x, u) and Q.is_polynomial(x, u)) or Q == 0:
        return None
    name = f.func.__name__
    taken = {s.name for s in P.free_symbols | Q.free_symbols if isinstance(s, Symbol) and s != u}
    y: Symbol = Dummy(name) if name in taken else Symbol(name)
    return as_expr(P.xreplace({u: y})), as_expr(Q.xreplace({u: y})), x, y


# The Darboux polynomials

class _Unknowns:
    """The coefficients of ``f`` (``a``) and of its cofactor (``b``) left
    unknown in one chart, as polynomials over ``K`` in those symbols."""

    def __init__(self, field: _Field, f_terms: dict[_Monomial, Expr], g_terms: dict[_Monomial, Expr],
                 a: list[Symbol], b: list[Symbol]) -> None:
        self.field = field
        self.f_terms = f_terms
        self.g_terms = g_terms
        self.a = a
        self.b = b

    def equations(self) -> list[Poly]:
        """The coefficients of ``D f - g f`` as polynomials in the
        unknowns."""
        x, y, K = self.field.x, self.field.y, self.field.K
        f = Add(*[c*x**i*y**j for (i, j), c in self.f_terms.items()])
        g = Add(*[c*x**i*y**j for (i, j), c in self.g_terms.items()])
        expression = (self.field.Q.as_expr()*f.diff(x) + self.field.P.as_expr()*f.diff(y) - g*f).expand()
        gens = self.a + self.b
        try:
            in_xy = Poly(expression, x, y)
        except PolynomialError:
            return [Poly(1, *gens, domain=K)]
        equations: list[Poly] = []
        for _, coefficient in in_xy.terms():
            equation = _unknown_poly(as_expr(coefficient), gens, K)
            if not equation.is_zero:
                equations.append(equation)
        return equations


def _unknown_poly(e: Expr, gens: Sequence[Symbol], K: Domain) -> Poly:
    if gens:
        return Poly(e, *gens, domain=K)
    # no unknown left: a constant, as a polynomial in a dummy
    return Poly(e, Dummy('u'), domain=K)


def _solve_affine(rows: list[list[DomainElement]], rhs: list[DomainElement], n: int,
                  K: Domain) -> Optional[_Affine]:
    """The solutions of ``rows * v = rhs`` over the field ``K``, ``v`` of
    length ``n``: a particular solution, a basis of the nullspace and the
    free columns, or ``None`` when there is none."""
    if not rows:
        identity = [[K.one if k == i else K.zero for k in range(n)] for i in range(n)]
        return [K.zero]*n, identity, list(range(n))
    augmented = DomainMatrix([row + [value] for row, value in zip(rows, rhs)], (len(rows), n + 1), K)
    reduced, pivots = augmented.rref()
    pivot_columns: list[int] = list(pivots)
    if n in pivot_columns:
        return None
    entries = reduced.to_list()
    particular: list[DomainElement] = [K.zero]*n
    for r, column in enumerate(pivot_columns):
        particular[column] = entries[r][n]
    free = [column for column in range(n) if column not in pivot_columns]
    basis: list[list[DomainElement]] = []
    for column in free:
        vector: list[DomainElement] = [K.zero]*n
        vector[column] = K.one
        for r, pivot in enumerate(pivot_columns):
            vector[pivot] = -entries[r][column]
        basis.append(vector)
    return particular, basis, free


def _linear_rows(equations: Sequence[Poly], gens: Sequence[Symbol],
                 K: Domain) -> tuple[list[list[DomainElement]], list[DomainElement]]:
    """The rows and right-hand sides of the equations of degree at most
    one in the unknowns."""
    rows: list[list[DomainElement]] = []
    rhs: list[DomainElement] = []
    index = {gen: i for i, gen in enumerate(gens)}
    for equation in equations:
        if equation.total_degree() > 1:
            continue
        row: list[DomainElement] = [K.zero]*len(gens)
        constant: DomainElement = K.zero
        # ``terms`` gives the coefficients as expressions: back to the
        # domain (over an algebraic field the matrix got ``One`` entries)
        for monomial, coefficient in equation.terms():
            if sum(monomial) == 0:
                constant = K.from_sympy(coefficient)
            else:
                position = monomial.index(1)
                row[index[equation.gens[position]]] = K.from_sympy(coefficient)
        rows.append(row)
        rhs.append(-constant)
    return rows, rhs


class _Propagation:
    """The linear equations of a bilinear system solved and substituted
    until only nonlinear ones remain: the values of the solved unknowns
    as affine expressions in the free ones."""

    def __init__(self, equations: list[Poly], a: list[Symbol], b: list[Symbol], K: Domain) -> None:
        self.K = K
        self.a = list(a)
        self.b = list(b)
        self.gens: list[Symbol] = self.a + self.b
        self.values: dict[Symbol, Expr] = {}
        self.equations = equations
        self.consistent = True
        self._run()

    def _run(self) -> None:
        while self.consistent:
            rows, rhs = _linear_rows(self.equations, self.gens, self.K)
            if not rows:
                return
            solved = _solve_affine(rows, rhs, len(self.gens), self.K)
            if solved is None:
                self.consistent = False
                return
            particular, basis, free = solved
            substitution: dict[Symbol, Expr] = {}
            for i, gen in enumerate(self.gens):
                if i in free:
                    continue
                value = as_expr(self.K.to_sympy(particular[i]))
                for vector, column in zip(basis, free):
                    if vector[i] != self.K.zero:
                        value = as_expr(value + self.K.to_sympy(vector[i])*self.gens[column])
                substitution[gen] = value
            if not substitution:
                return
            self._substitute(substitution)

    def _substitute(self, substitution: dict[Symbol, Expr]) -> None:
        for gen, value in self.values.items():
            self.values[gen] = as_expr(value.xreplace(substitution))
        self.values.update(substitution)
        self.gens = [gen for gen in self.gens if gen not in substitution]
        self.a = [gen for gen in self.a if gen not in substitution]
        self.b = [gen for gen in self.b if gen not in substitution]
        remaining: list[Poly] = []
        for equation in self.equations:
            replaced = _unknown_poly(as_expr(equation.as_expr().xreplace(substitution)).expand(), self.gens, self.K)
            if replaced.is_zero:
                continue
            if not self.gens or replaced.total_degree() == 0:
                # a nonzero constant: no solution
                self.consistent = False
                return
            remaining.append(replaced)
        self.equations = remaining

    def value_of(self, gen: Symbol, point: dict[Symbol, Expr]) -> Expr:
        """The value of an unknown at a point of the free unknowns."""
        if gen in self.values:
            return as_expr(self.values[gen].xreplace(point))
        return point[gen]


def _chart_equations(field: _Field, N: int, top: Poly, top_cofactor: Poly,
                     fixed: Optional[dict[_Monomial, Expr]] = None) -> _Unknowns:
    """The unknown coefficients of a Darboux polynomial of degree ``N``
    whose highest degree part is ``top`` (or, with ``fixed``, whose
    highest degree coefficients are partly fixed and partly unknown) and
    whose cofactor has highest degree part ``top_cofactor``."""
    x, y = field.x, field.y
    a: list[Symbol] = []
    b: list[Symbol] = []
    f_terms: dict[_Monomial, Expr] = {}
    for monomial in _monomials(N):
        if sum(monomial) == N:
            if fixed is not None and monomial in fixed:
                f_terms[monomial] = fixed[monomial]
                continue
            if fixed is None:
                f_terms[monomial] = as_expr(top.coeff_monomial(x**monomial[0]*y**monomial[1]))
                continue
        symbol = Dummy('a_%d_%d' % monomial)
        a.append(symbol)
        f_terms[monomial] = symbol
    g_terms: dict[_Monomial, Expr] = {}
    for monomial in _monomials(field.degree - 1):
        if sum(monomial) == field.degree - 1:
            g_terms[monomial] = as_expr(top_cofactor.coeff_monomial(x**monomial[0]*y**monomial[1]))
            continue
        symbol = Dummy('b_%d_%d' % monomial)
        b.append(symbol)
        g_terms[monomial] = symbol
    return _Unknowns(field, f_terms, g_terms, a, b)


def _share_of_time() -> Optional[float]:
    """The time given to one optional step (a polynomial system of a
    leaf, a simplification): a share of the enclosing limit, or of the
    default."""
    left = remaining_time()
    limit = settings.timeout if left is None else left
    return None if limit is None else limit/4


def _in_domain(value: Expr, K: Domain) -> Optional[DomainElement]:
    try:
        return K.from_sympy(value)
    except (CoercionFailed, TypeError, ValueError):
        return None


#: a row of a linear system in the polynomial unknowns whose entries are
#: polynomials in the cofactor unknowns: the coefficients of the unknowns
#: and, last, the constant term (``sum c_i a_i + r = 0``)
_Row = list[PolyElement]
#: the constraints on the cofactor unknowns at a leaf of the parametric
#: elimination: polynomials which vanish, polynomials which do not, and
#: the unknowns fixed by the vanishing pivots of degree one (their
#: index and their value in the unknowns fixed later and the free ones)
_Leaf = tuple[list[PolyElement], list[PolyElement], list[tuple[int, PolyElement]]]


def _degree(p: PolyElement) -> int:
    return max((sum(monomial) for monomial in p.monoms()), default=-1)


class _ParametricElimination:
    """Gaussian elimination of a linear system in the polynomial unknowns
    whose coefficients are polynomials in the cofactor unknowns (the
    parameters), with a case split at every pivot which may vanish: the
    leaves carry the conditions on the parameters under which the system
    is consistent (the constant rows left, the pivots assumed zero) and
    the pivots assumed nonzero. A pivot which is a nonzero constant needs
    no split and is preferred anywhere in the matrix; a vanishing pivot
    of degree one eliminates a parameter by substitution."""

    #: the number of leaves after which the elimination is given up
    LIMIT = 400

    def __init__(self, ring: PolyRing, n: int) -> None:
        self.ring = ring
        self.n = n
        self.leaves: list[_Leaf] = []
        self.exceeded = False

    def run(self, rows: list[_Row], columns: list[int], equations: list[PolyElement],
            nonzero: list[PolyElement], fixed: list[tuple[int, PolyElement]]) -> None:
        if self.exceeded:
            return
        live: list[_Row] = []
        # after a substitution an equation may have become zero (dropped)
        # or a nonzero constant (no cofactor on this branch)
        equations = [e for e in equations if e != 0]
        if any(e.is_ground for e in equations):
            return
        for row in rows:
            if all(entry == 0 for entry in row[:self.n]):
                constant = row[self.n]
                if constant != 0:
                    if constant.is_ground:
                        return
                    equations.append(constant)
            else:
                live.append(row)
        if not live or not columns:
            self.leaves.append((equations, nonzero, fixed))
            if len(self.leaves) > self.LIMIT:
                self.exceeded = True
            return
        pivot = self._constant_pivot(live, columns)
        if pivot is not None:
            i, j = pivot
            self.run(self._eliminated(live, i, j, nonzero), [c for c in columns if c != j], equations, nonzero, fixed)
            return
        i, j = self._smallest_pivot(live, columns)
        p = live[i][j]
        # the pivot vanishes
        substitution = self._substitution(p)
        if substitution is not None:
            index, value = substitution
            gen = self.ring.gens[index]
            self.run([[entry.compose(gen, value) for entry in row] for row in live], columns,
                     [e.compose(gen, value) for e in equations], [e.compose(gen, value) for e in nonzero],
                     fixed + [(index, value)])
        else:
            zeroed = [list(row) for row in live]
            zeroed[i][j] = self.ring.zero
            self.run(zeroed, columns, equations + [p], nonzero, fixed)
        # the pivot does not vanish
        self.run(self._eliminated(live, i, j, nonzero + [p]), [c for c in columns if c != j], equations,
                 nonzero + [p], fixed)

    def _constant_pivot(self, rows: list[_Row], columns: list[int]) -> Optional[tuple[int, int]]:
        for i, row in enumerate(rows):
            for j in columns:
                entry = row[j]
                if entry != 0 and entry.is_ground:
                    return i, j
        return None

    def _smallest_pivot(self, rows: list[_Row], columns: list[int]) -> tuple[int, int]:
        best: Optional[tuple[int, int, int]] = None
        for i, row in enumerate(rows):
            for j in columns:
                entry = row[j]
                if entry != 0:
                    degree = _degree(entry)
                    if best is None or degree < best[0]:
                        best = (degree, i, j)
        if best is None:
            raise ValueError("no pivot")
        return best[1], best[2]

    def _substitution(self, p: PolyElement) -> Optional[tuple[int, PolyElement]]:
        """``(k, value)`` with ``p = 0`` solved for the parameter ``b_k``
        when ``p`` has degree one."""
        if _degree(p) != 1:
            return None
        for index, gen in enumerate(self.ring.gens):
            coefficient = p.coeff(gen)
            if coefficient != 0:
                rest = p - gen.mul_ground(coefficient)
                return index, rest.quo_ground(-coefficient)
        return None

    def _eliminated(self, rows: list[_Row], i: int, j: int, nonzero: list[PolyElement]) -> list[_Row]:
        """The rows with the unknown ``j`` eliminated by the pivot row
        ``i`` (fraction-free), the pivot row dropped, and the factors
        known to be nonzero divided out of each row."""
        pivot_row = rows[i]
        p = pivot_row[j]
        result: list[_Row] = []
        for k, row in enumerate(rows):
            if k == i:
                continue
            c = row[j]
            if c == 0:
                result.append(row)
                continue
            combined = [entry*p - pivot_entry*c for entry, pivot_entry in zip(row, pivot_row)]
            result.append(self._reduced(combined, nonzero))
        return result

    def _reduced(self, row: _Row, nonzero: list[PolyElement]) -> _Row:
        """The row divided by the factors common to its entries which
        are known to be nonzero (the earlier pivots, as in Bareiss'
        elimination). A common factor which may vanish stays: dividing
        it out loses the cofactors at which it does (the bug: a leaf
        ``b**3 = 0`` became ``1 = 0``, and the polynomial ``x`` of ``y' =
        (x y - 1)/x**2`` was lost)."""
        entries = [entry for entry in row if entry != 0]
        if not entries:
            return row
        content = entries[0]
        for entry in entries[1:]:
            content = content.gcd(entry)
            if content.is_ground:
                return row
        for factor in nonzero:
            if factor.is_ground:
                continue
            while True:
                quotient, remainder = content.div(factor)
                if remainder != 0 or quotient == 0:
                    break
                content = quotient
                row = [entry.exquo(factor) for entry in row]
        return row


def _cofactor_points(propagation: _Propagation) -> Optional[list[dict[Symbol, Expr]]]:
    """The values of the free cofactor unknowns on the solutions of the
    nonlinear remainder, which is linear in the polynomial unknowns with
    coefficients polynomial in the cofactor unknowns: the parametric
    elimination gives the conditions on the cofactor unknowns at its
    leaves, zero-dimensional systems (the cofactors of bounded degree are
    finitely many) solved one by one; ``None`` when the elimination or a
    system could not be finished."""
    K = propagation.K
    a, b = propagation.a, propagation.b
    if not b:
        return [{}]
    ring = PolyRing(tuple(b), K)
    domain: Domain = ring.to_domain()
    n = len(a)
    rows: list[_Row] = []
    constraints: list[PolyElement] = []
    for equation in propagation.equations:
        if not a:
            constraints.append(ring.from_expr(equation.as_expr()))
            continue
        in_a = Poly(equation.as_expr(), *a, domain=domain)
        row: _Row = [ring.zero]*(n + 1)
        for monomial, coefficient in in_a.as_dict(native=True).items():
            if sum(monomial) == 0:
                row[n] = coefficient
            elif sum(monomial) == 1:
                row[monomial.index(1)] = coefficient
            else:
                return None
        rows.append(row)
    elimination = _ParametricElimination(ring, n)
    elimination.run(rows, list(range(n)), constraints, [], [])
    if elimination.exceeded:
        return None
    points: list[dict[Symbol, Expr]] = []
    seen: set[tuple[Expr, ...]] = set()
    for equations, nonzero, fixed in elimination.leaves:
        fixed_indices = {index for index, _ in fixed}
        free = [index for index in range(len(b)) if index not in fixed_indices]
        if not equations and free:
            # a continuum of cofactors: not for a field of bounded degree
            continue
        for values in _leaf_points(equations, [b[index] for index in free], K):
            point: dict[int, DomainElement] = dict(zip(free, values))
            # the fixed unknowns, the last fixed first (its value is in
            # the free ones), then the earlier ones
            for index, value in reversed(fixed):
                point[index] = _at(value, [point[k] for k in sorted(point)], ring, sorted(point))
            full = [point[index] for index in range(len(b))]
            if any(_at(e, full, ring) == K.zero for e in nonzero):
                continue
            key = tuple(as_expr(K.to_sympy(v)) for v in full)
            if key in seen:
                continue
            seen.add(key)
            points.append(dict(zip(b, key)))
    return points


def _at(e: PolyElement, values: list[DomainElement], ring: PolyRing,
        indices: Optional[list[int]] = None) -> DomainElement:
    """``e`` at the values of the unknowns (of the unknowns ``indices``,
    the only ones it may involve, when given)."""
    if e.is_ground:
        return e.coeff(1)
    gens = ring.gens if indices is None else [ring.gens[k] for k in indices]
    value = e.evaluate(list(zip(gens, values)))
    if isinstance(value, PolyElement):
        if not value.is_ground:
            raise ValueError("an unknown is left in a fixed value")
        return value.coeff(1)
    return value


def _leaf_points(equations: list[PolyElement], gens: list[Symbol], K: Domain) -> list[list[DomainElement]]:
    """The solutions in ``K`` of the equations of a leaf in its free
    unknowns ``gens`` (none when a system is not zero-dimensional or not
    solved in time). The equations are products of minors: each is
    factored, and the cases, one factor per equation, are solved as soon
    as the factors chosen form a zero-dimensional system, the remaining
    equations being checked at its points."""
    if not gens:
        return [[]] if all(e == 0 for e in equations) else []
    cases: list[list[Poly]] = []
    for e in equations:
        factors = _factors(as_expr(e.as_expr()), gens, K)
        if not factors:
            return []
        cases.append(factors)
    cases.sort(key=len)
    points = attempt(lambda: _points_of_cases(cases, [], gens, K), _share_of_time())
    return [] if points is None else points


def _factors(e: Expr, gens: list[Symbol], K: Domain) -> list[Poly]:
    """The monic irreducible factors of ``e`` over ``K``, without the
    constants."""
    _, factors = Poly(e, *gens, domain=K).factor_list()
    return [as_poly.monic() for as_poly, _ in factors if as_poly.total_degree() > 0]


def _points_of_cases(cases: list[list[Poly]], chosen: list[Poly], gens: list[Symbol],
                     K: Domain) -> list[list[DomainElement]]:
    """The points of the cases, one factor of each equation of ``cases``
    added to ``chosen`` in turn."""
    if len(chosen) >= len(gens):
        points = _points_in_domain([as_expr(p.as_expr()) for p in chosen], gens, K)
        if points is not None:
            # the remaining equations hold at the point when a factor does
            return [point for point in points
                    if all(any(_evaluated_poly(f, point, K) == K.zero for f in factors) for factors in cases)]
    if not cases:
        points = _points_in_domain([as_expr(p.as_expr()) for p in chosen], gens, K)
        return [] if points is None else points
    found: list[list[DomainElement]] = []
    seen: set[tuple[Expr, ...]] = set()
    for factor in cases[0]:
        for point in _points_of_cases(cases[1:], chosen + [factor], gens, K):
            key = tuple(as_expr(K.to_sympy(v)) for v in point)
            if key not in seen:
                seen.add(key)
                found.append(point)
    return found


def _evaluated_poly(p: Poly, point: list[DomainElement], K: Domain) -> DomainElement:
    return K.convert(p.eval(dict(zip(p.gens, [K.to_sympy(v) for v in point]))))


def _lex_basis(exprs: list[Expr], gens: list[Symbol], K: Domain) -> Optional[list[Expr]]:
    """The reduced lex Gröbner basis of a zero-dimensional system, from
    the grevlex basis by FGLM; ``None`` when the system is not
    zero-dimensional (its grevlex basis is far cheaper than a lex one)."""
    from sympy.polys.polytools import groebner
    basis = groebner(exprs, *gens, order='grevlex', domain=K)
    if list(basis.exprs) == [S.One]:
        return [S.One]
    if not basis.is_zero_dimensional:
        return None
    return [as_expr(g) for g in basis.fglm('lex').exprs]


def _points_in_domain(exprs: list[Expr], gens: list[Symbol], K: Domain) -> Optional[list[list[DomainElement]]]:
    """The points with coordinates in ``K`` of a zero-dimensional system
    (``None`` when it is not zero-dimensional): the last coordinate is a
    root of a polynomial of the system in the last unknown alone (the
    gcd of those; from the lex Gröbner basis when the system has none),
    and the rest is solved recursively at each root, an inconsistent
    remainder giving no point."""
    exprs = [e for e in exprs if e != 0]
    if not gens:
        return [[]] if not exprs else []
    if not exprs:
        return None
    if any(e.is_number for e in exprs):
        return []
    last = gens[-1]
    others = set(gens[:-1])
    univariate = [e for e in exprs if not (e.free_symbols & others)]
    if not univariate:
        basis = _lex_basis(exprs, gens, K)
        if basis is None:
            return None
        if basis == [S.One]:
            return []
        exprs = basis
        univariate = [e for e in exprs if not (e.free_symbols & others)]
        if not univariate:
            return None
    polynomial = Poly(univariate[0], last, domain=K)
    for e in univariate[1:]:
        polynomial = polynomial.gcd(Poly(e, last, domain=K))
    points: list[list[DomainElement]] = []
    for root in polynomial.ground_roots():
        value = _in_domain(as_expr(root), K)
        if value is None:
            continue
        rest = [as_expr(e.xreplace({last: as_expr(root)}).expand()) for e in exprs if e not in univariate]
        below = _points_in_domain(rest, gens[:-1], K)
        if below is None:
            return None
        for point in below:
            points.append(point + [value])
    return points


def _members(particular: list[DomainElement], basis: list[list[DomainElement]], K: Domain) -> list[list[DomainElement]]:
    """A few members of an affine space: the sparsest member reachable
    from the particular solution along the basis (the member of a pencil
    ``f + u h`` with the fewest terms, ``x**2 - y**2 + 1`` rather than
    ``x**2 + 2 x - y**2 + 1`` when ``x`` is a Darboux polynomial with the
    same cofactor), each basis vector added to it, and a generic
    combination."""
    sparse = _sparse_member(particular, basis, K)
    members = [sparse]
    for vector in basis:
        members.append([p + v for p, v in zip(sparse, vector)])
    if len(basis) > 1:
        generic = list(sparse)
        for k, vector in enumerate(basis):
            generic = [p + K.convert(k + 1)*v for p, v in zip(generic, vector)]
        members.append(generic)
    return members


def _sparse_member(particular: list[DomainElement], basis: list[list[DomainElement]],
                   K: Domain) -> list[DomainElement]:
    """The member with the fewest nonzero entries among those reached by
    cancelling one entry at a time along each basis vector (a greedy
    pass over the basis, twice)."""
    member = list(particular)
    for _ in range(2):
        for vector in basis:
            best = member
            for i, v in enumerate(vector):
                if v == K.zero or member[i] == K.zero:
                    continue
                u = -member[i]/v
                candidate = [p + u*w for p, w in zip(member, vector)]
                if _nonzero_entries(candidate) < _nonzero_entries(best):
                    best = candidate
            member = best
    return member


def _nonzero_entries(vector: list[DomainElement]) -> int:
    return sum(1 for v in vector if v != 0)


def _chart_solutions(unknowns: _Unknowns) -> list[tuple[Poly, Poly]]:
    """The Darboux polynomials of one chart with their cofactors, a few
    members of each affine family."""
    field = unknowns.field
    K = field.K
    propagation = _Propagation(unknowns.equations(), unknowns.a, unknowns.b, K)
    if not propagation.consistent:
        return []
    if propagation.equations:
        points = _cofactor_points(propagation)
        if points is None:
            return []
    else:
        points = [{}]
    found: list[tuple[Poly, Poly]] = []
    for point in points:
        # at a cofactor the remaining equations are linear in the
        # polynomial unknowns
        free = [gen for gen in propagation.a if gen not in point]
        equations = [_unknown_poly(as_expr(equation.as_expr().xreplace(point)).expand(), free, K)
                     for equation in propagation.equations]
        equations = [equation for equation in equations if not equation.is_zero]
        if any(equation.total_degree() == 0 for equation in equations):
            continue
        if any(equation.total_degree() > 1 for equation in equations):
            continue
        rows, rhs = _linear_rows(equations, free, K)
        solved = _solve_affine(rows, rhs, len(free), K)
        if solved is None:
            continue
        particular, basis, _ = solved
        for member in _members(particular, basis, K):
            values = dict(point)
            values.update({gen: as_expr(K.to_sympy(value)) for gen, value in zip(free, member)})
            # the free cofactor unknowns left by the propagation but absent
            # from the point (none unless the system was linear)
            for gen in propagation.b:
                values.setdefault(gen, S.Zero)
            f_expr = Add(*[c*field.x**i*field.y**j
                           for (i, j), c in _evaluated(unknowns.f_terms, propagation, values).items()])
            g_expr = Add(*[c*field.x**i*field.y**j
                           for (i, j), c in _evaluated(unknowns.g_terms, propagation, values).items()])
            f = _poly(f_expr, field)
            g = _poly(g_expr, field)
            if f.is_zero or _total_degree(f) == 0:
                continue
            if field.apply(f) != g*f:
                continue
            found.append((f, g))
    return found


def _evaluated(terms: dict[_Monomial, Expr], propagation: _Propagation,
               values: dict[Symbol, Expr]) -> dict[_Monomial, Expr]:
    result: dict[_Monomial, Expr] = {}
    for monomial, coefficient in terms.items():
        if isinstance(coefficient, Symbol) and coefficient in propagation.values:
            result[monomial] = propagation.value_of(coefficient, values)
        else:
            result[monomial] = as_expr(coefficient.xreplace(values))
    return result


def _top_parts(field: _Field) -> tuple[Poly, Poly, Poly]:
    """``P_d``, ``Q_d`` and ``H = x P_d - y Q_d``."""
    d = field.degree
    P_d = field.homogeneous_part(field.P, d)
    Q_d = field.homogeneous_part(field.Q, d)
    H = _poly(field.x*P_d.as_expr() - field.y*Q_d.as_expr(), field)
    return P_d, Q_d, H


def _top_candidates(field: _Field, N: int, H: Poly) -> list[Poly]:
    """The possible highest degree parts of a Darboux polynomial of
    degree ``N``: the monic products of irreducible factors of ``H`` of
    total degree ``N``."""
    _, factors = H.factor_list()
    irreducible: list[tuple[Poly, int]] = [(_poly(h, field).monic(), _total_degree(h)) for h, _ in factors]
    candidates: list[Poly] = []
    seen: set[Expr] = set()

    def extend(index: int, degree_left: int, product: Poly) -> None:
        if degree_left == 0:
            key = as_expr(product.as_expr())
            if key not in seen:
                seen.add(key)
                candidates.append(product)
            return
        for k in range(index, len(irreducible)):
            h, degree = irreducible[k]
            if degree <= degree_left:
                extend(k, degree_left - degree, product*h)

    extend(0, N, field.one())
    return candidates


def _darboux_of_degree(field: _Field, N: int) -> list[tuple[Poly, Poly]]:
    """Darboux polynomials of total degree ``N`` with their cofactors (a
    few members of each family, not factored)."""
    P_d, Q_d, H = _top_parts(field)
    x, y = field.x, field.y
    found: list[tuple[Poly, Poly]] = []
    if not H.is_zero:
        for top in _top_candidates(field, N, H):
            D_top = _poly(Q_d*top.diff(x) + P_d*top.diff(y), field)
            quotient, remainder = D_top.div(top)
            if not remainder.is_zero:
                continue
            found.extend(_chart_solutions(_chart_equations(field, N, top, _poly(quotient, field))))
        return found
    # dicritical: D_d = k (x d/dx + y d/dy), the cofactor's top part is N k
    k = Q_d.div(_poly(x, field))[0] if not Q_d.is_zero else P_d.div(_poly(y, field))[0]
    k = _poly(k, field)
    top_cofactor = _poly(N*k.as_expr(), field)
    top_monomials = [(N - j, j) for j in range(N + 1)]
    for leading in range(N + 1):
        fixed: dict[_Monomial, Expr] = {}
        for j in range(leading):
            fixed[top_monomials[j]] = S.Zero
        fixed[top_monomials[leading]] = S.One
        found.extend(_chart_solutions(_chart_equations(field, N, field.zero(), top_cofactor, fixed)))
    return found


def _irreducible_factors(field: _Field, f: Poly) -> list[Poly]:
    _, factors = f.factor_list()
    return [_poly(h, field).monic() for h, _ in factors if _total_degree(h) > 0]


def _darboux(field: _Field, degree: int) -> list[tuple[Poly, Poly]]:
    """The irreducible Darboux polynomials of degree at most ``degree``
    with their cofactors, monic, without repetition."""
    found: dict[Expr, tuple[Poly, Poly]] = {}
    for N in range(1, degree + 1):
        _add_darboux(field, N, found)
    return list(found.values())


def _field_of(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, extension: Optional[Expr] = None) -> _Field:
    p, q = _coefficient_domain(as_expr(P), as_expr(Q), x, y, extension)
    if q.is_zero:
        raise ValueError("Q must not be zero")
    return _Field(p, q, x, y)


def darboux_polynomials(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, degree: int = 4,
                        extension: Optional[Expr] = None) -> list[DarbouxPolynomial]:
    """The irreducible Darboux polynomials of total degree at most
    ``degree`` of the field ``Q d/dx + P d/dy`` (the equation ``y' = P/Q``)
    with their cofactors, over the field generated by the coefficients
    (and ``extension``, an algebraic number such as ``I`` or ``sqrt(2)``,
    when given: ``x + I*y`` and ``x - I*y`` rather than their product).

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.darboux import darboux_polynomials
    >>> for f in darboux_polynomials(y*(x - 1), x*(1 - y), x, y):      # Lotka–Volterra
    ...     print(f)
    DarbouxPolynomial(y, x - 1)
    DarbouxPolynomial(x, 1 - y)
    >>> darboux_polynomials(x + y, x - y, x, y, degree=2)
    [DarbouxPolynomial(x**2 + y**2, 2)]
    """
    field = _field_of(P, Q, x, y, extension)
    return [DarbouxPolynomial(as_expr(f.as_expr()), as_expr(g.as_expr())) for f, g in _darboux(field, degree)]


# Exponential factors

def _denominators(field: _Field, polynomials: Sequence[tuple[Poly, Poly]],
                  degree: int) -> list[tuple[Poly, Poly]]:
    """``1`` and the products of Darboux polynomials of total degree at
    most ``degree``, with their cofactors."""
    result: list[tuple[Poly, Poly]] = [(field.one(), field.zero())]
    degrees = [_total_degree(f) for f, _ in polynomials]
    for size in range(1, degree + 1):
        for choice in combinations_with_replacement(range(len(polynomials)), size):
            if sum(degrees[i] for i in choice) > degree:
                continue
            B, g = field.one(), field.zero()
            for i in choice:
                B, g = B*polynomials[i][0], g + polynomials[i][1]
            result.append((_poly(B, field), _poly(g, field)))
    return result


def _native_terms(p: Poly, K: Domain) -> dict[_Monomial, DomainElement]:
    """The terms of a polynomial in ``x, y`` with the coefficients as
    elements of ``K``."""
    return {(monomial[0], monomial[1]): K.from_sympy(coefficient) for monomial, coefficient in p.terms()}


def _exponential_with_denominator(field: _Field, B: Poly, g_B: Poly, bound: int) -> list[tuple[Poly, Poly]]:
    """The exponential factors ``exp(A/B)`` with ``deg A <= bound`` and
    their cofactors ``L``: the nullspace of ``(A, L) -> D A - g_B A - L B``,
    without those with ``L = 0`` (rational first integrals or constants)."""
    x, y, K = field.x, field.y, field.K
    a_monomials = _monomials(bound)
    l_monomials = _monomials(field.degree - 1)
    # the columns of the linear map: the images of the monomials of A
    # and of L, as coefficient vectors on the monomials of the image
    columns: list[dict[_Monomial, DomainElement]] = []
    for i, j in a_monomials:
        monomial = Poly.from_dict({(i, j): K.one}, x, y, domain=K)
        columns.append(_native_terms(field.apply(monomial) - g_B*monomial, K))
    for i, j in l_monomials:
        monomial = Poly.from_dict({(i, j): K.one}, x, y, domain=K)
        columns.append(_native_terms(-B*monomial, K))
    image: list[_Monomial] = sorted({monomial for column in columns for monomial in column})
    rows: list[list[DomainElement]] = [[column.get(monomial, K.zero) for column in columns] for monomial in image]
    solved = _solve_affine(rows, [K.zero]*len(rows), len(columns), K)
    if solved is None:
        return []
    _, basis, _ = solved
    result: list[tuple[Poly, Poly]] = []
    for vector in basis:
        A_poly = Poly.from_dict({m: vector[i] for i, m in enumerate(a_monomials)}, x, y, domain=K)
        L_poly = Poly.from_dict({m: vector[len(a_monomials) + i] for i, m in enumerate(l_monomials)}, x, y, domain=K)
        if L_poly.is_zero or A_poly.is_zero:
            continue
        result.append((A_poly, L_poly))
    return result


def _exponential(field: _Field, polynomials: Sequence[tuple[Poly, Poly]],
                 degree: int) -> list[tuple[Poly, Poly, Poly]]:
    """The exponential factors ``(A, B, L)`` with ``B`` a product of the
    Darboux polynomials of degree at most ``degree``."""
    found: list[tuple[Poly, Poly, Poly]] = []
    seen: set[Expr] = set()
    for B, g_B in _denominators(field, polynomials, degree):
        bound = max(degree, field.degree) + _total_degree(B)
        for A, L in _exponential_with_denominator(field, B, g_B, bound):
            # exp(B A0/B) is exp(A0) again
            key = as_expr(cancel(A.as_expr()/B.as_expr()))
            if key in seen:
                continue
            seen.add(key)
            found.append((A, B, L))
    return found


def exponential_factors(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, degree: int = 4,
                        extension: Optional[Expr] = None) -> list[ExponentialFactor]:
    """The exponential factors ``exp(A/B)`` of the field ``Q d/dx + P d/dy``
    whose denominators are products of its Darboux polynomials of degree
    at most ``degree`` (``1`` included) and whose numerators have degree
    at most ``max(degree, d) + deg B``, with their cofactors ``D(A/B)``.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.darboux import exponential_factors
    >>> exponential_factors(x*y + 1, 1, x, y, degree=1)         # y' = x y + 1
    [ExponentialFactor(x, 1, 1), ExponentialFactor(x**2/2, 1, x)]
    """
    field = _field_of(P, Q, x, y, extension)
    polynomials = _darboux(field, degree)
    return [ExponentialFactor(as_expr(A.as_expr()), as_expr(B.as_expr()), as_expr(L.as_expr()))
            for A, B, L in _exponential(field, polynomials, degree)]


# The exponents

def _cofactor_rows(field: _Field, cofactors: Sequence[Poly]) -> tuple[list[list[DomainElement]], list[_Monomial]]:
    """The matrix whose columns are the coefficient vectors of the
    cofactors on the monomials of degree at most ``d - 1``."""
    monomials = _monomials(field.degree - 1)
    rows: list[list[DomainElement]] = []
    x, y = field.x, field.y
    for i, j in monomials:
        rows.append([field.K.convert(g.coeff_monomial(x**i*y**j)) for g in cofactors])
    return rows, monomials


def _integers(vector: list[DomainElement], K: Domain) -> Optional[list[Expr]]:
    """The vector scaled to integers when its entries are rational."""
    values = [as_expr(K.to_sympy(v)) for v in vector]
    rationals = [v for v in values if isinstance(v, Rational)]
    if len(rationals) != len(values):
        return None
    scale = 1
    for v in rationals:
        scale = lcm(scale, int(v.q))
    return [as_expr(v*scale) for v in rationals]


class _Darboux:
    """The Darboux polynomials of degree at most ``degree`` (given) and
    the exponential factors of a field with the linear algebra of their
    exponents."""

    def __init__(self, field: _Field, polynomials: list[tuple[Poly, Poly]], degree: int) -> None:
        self.field = field
        self.polynomials = polynomials
        self.exponentials = _exponential(field, self.polynomials, degree)
        self.cofactors: list[Poly] = [g for _, g in self.polynomials] + [L for _, _, L in self.exponentials]
        self.rows, self.monomials = _cofactor_rows(field, self.cofactors)

    def first_integral(self) -> Optional[Expr]:
        """``sum c_i log f_i + sum e_j A_j/B_j`` with ``sum c_i g_i + sum
        e_j L_j = 0``, written ``prod f_i**c_i`` (a rational first
        integral) when no exponential factor takes part, the exponents
        integers when they are rational, the simplest combination among
        a basis; ``None`` when the cofactors are independent."""
        if not self.cofactors:
            return None
        K = self.field.K
        solved = _solve_affine(self.rows, [K.zero]*len(self.rows), len(self.cofactors), K)
        if solved is None:
            return None
        _, basis, _ = solved
        if not basis:
            return None
        candidates: list[Expr] = []
        for vector in basis:
            scaled = _integers(vector, K)
            exponents = scaled if scaled is not None else [as_expr(K.to_sympy(v)) for v in vector]
            # the first exponent positive (a sign fixed, for a stable form)
            first = next((e for e in exponents if e != 0), S.One)
            if first.is_negative:
                exponents = [as_expr(-e) for e in exponents]
            candidate = self._product(exponents, logarithmic=True)
            if not _is_constant_integral(_tidy(candidate), self.field.x, self.field.y):
                candidates.append(candidate)
        if not candidates:
            return None
        return min(candidates, key=_simplicity)

    def integrating_factor(self) -> Optional[Expr]:
        """``prod f_i**c_i exp(sum e_j A_j/B_j)`` with ``sum c_i g_i + sum
        e_j L_j = -div D``, the simplest among the particular solution and
        its sums with the basis of the homogeneous solutions."""
        K = self.field.K
        divergence = self.field.divergence()
        x, y = self.field.x, self.field.y
        rhs = [-K.convert(divergence.coeff_monomial(x**i*y**j)) for i, j in self.monomials]
        if not self.cofactors:
            if all(v == K.zero for v in rhs):
                return S.One
            return None
        solved = _solve_affine(self.rows, rhs, len(self.cofactors), K)
        if solved is None:
            return None
        particular, basis, _ = solved
        candidates = [self._product([as_expr(K.to_sympy(v)) for v in member], logarithmic=False)
                      for member in [particular] + _members(particular, basis, K)]
        return min(candidates, key=_simplicity)

    def _product(self, exponents: Sequence[Expr], logarithmic: bool) -> Expr:
        """``prod f_i**c_i exp(sum e_j A_j/B_j)``, or with ``logarithmic``
        its logarithm when an exponential factor takes part."""
        n = len(self.polynomials)
        argument: Expr = S.Zero
        for (A, B, _), e in zip(self.exponentials, exponents[n:]):
            if e != 0:
                argument = as_expr(argument + e*A.as_expr()/B.as_expr())
        if logarithmic and argument != 0:
            terms: list[Expr] = [argument]
            for (f, _), c in zip(self.polynomials, exponents[:n]):
                if c != 0:
                    terms.append(as_expr(c*log(f.as_expr())))
            return as_expr(Add(*terms))
        factors: list[Expr] = []
        for (f, _), c in zip(self.polynomials, exponents[:n]):
            if c != 0:
                factors.append(as_expr(f.as_expr()**c))
        if argument != 0:
            factors.append(as_expr(exp(argument)))
        return as_expr(Mul(*factors))


def _simplicity(e: Expr) -> tuple[int, int, int, str]:
    """The sort key of the simplest candidate: the exponentials first,
    then the powers with a non-integer exponent, then the operations,
    then the printed form (for a deterministic choice)."""
    from sympy.core.power import Pow
    fractional = sum(1 for p in e.atoms(Pow) if not p.exp.is_Integer)
    return len(e.atoms(exp)), fractional, int(count_ops(e)), str(e)


def darboux_integrating_factor(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, degree: int = 4,
                               extension: Optional[Expr] = None) -> Optional[Expr]:
    """A Darboux integrating factor ``R`` of ``Q dy - P dx`` (``R (Q dy -
    P dx)`` is exact): a product of powers of the Darboux polynomials of
    degree at most ``degree`` and of exponential factors; ``None`` when
    the exponents have no solution.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.darboux import darboux_integrating_factor
    >>> darboux_integrating_factor(y*(x - 1), x*(1 - y), x, y)      # Lotka–Volterra
    1/(x*y)
    >>> darboux_integrating_factor(x*y + 1, 1, x, y, degree=1)      # y' = x y + 1
    exp(-x**2/2)
    """
    field = _field_of(P, Q, x, y, extension)
    return _Darboux(field, _darboux(field, degree), degree).integrating_factor()


def _integrate(e: Expr, v: Symbol, x: Symbol, y: Symbol) -> Optional[Expr]:
    """``Integral(e, v)`` by SymPy within half of the time left, with
    ``x``, ``y`` and the parameters real first (the logarithms of a
    rational integrand come out real: ``atan((x + y)/a)`` rather than
    ``I*log(x + y - I*a)/2 - I*log(x + y + I*a)/2``), then with complex
    symbols when that gives no antiderivative: a ``Piecewise`` on the
    signs of the parameters without a generic branch, or a wrong one
    (SymPy's ``integrate(a/(a*y**2 - b), y)`` is ``0`` for real ``a``,
    ``b``, ``y``; every antiderivative is checked by differentiation);
    ``None`` when neither is an antiderivative in closed form."""
    from sympy.integrals.integrals import integrate
    real = {s: Dummy(s.name, real=True) for s in e.free_symbols if isinstance(s, Symbol)}
    real.setdefault(x, Dummy(x.name, real=True))
    real.setdefault(y, Dummy(y.name, real=True))
    back = {dummy: symbol for symbol, dummy in real.items()}
    limit = remaining_time()
    budget = settings.timeout if limit is None else limit/2
    result = attempt(lambda: integrate(e.xreplace(real), real[v], conds='none'), budget)
    if result is not None:
        generic = _generic_branch(as_expr(result))
        if generic is not None:
            candidate = as_expr(generic.xreplace(back))
            if _is_antiderivative(candidate, e, v):
                return candidate
    limit = remaining_time()
    budget = settings.timeout if limit is None else limit/2
    result = attempt(lambda: integrate(e, v, conds='none'), budget)
    if result is None:
        return None
    generic = _generic_branch(as_expr(result))
    if generic is None or not _is_antiderivative(generic, e, v):
        return None
    return generic


def _is_antiderivative(F: Expr, e: Expr, v: Symbol) -> bool:
    """Whether ``F`` is an antiderivative of ``e`` in closed form: no
    integral or derivative left, and ``F' - e`` zero."""
    if F.has(Derivative) or _has_integral(F):
        return False
    return _is_zero(as_expr(F.diff(v) - e))


def _generic_branch(e: Expr) -> Optional[Expr]:
    """``e`` with every ``Piecewise`` replaced by its branch holding
    generically (a condition which is an inequation, or ``True``); the
    result is checked by differentiation by the caller."""
    from sympy.core.relational import Ne
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.functions.elementary.piecewise import ExprCondPair
    for piece in e.atoms(Piecewise):
        chosen: Optional[Expr] = None
        for pair in piece.args:
            if not isinstance(pair, ExprCondPair):
                continue
            if pair.cond == True or isinstance(pair.cond, Ne):
                chosen = as_expr(pair.expr)
                break
        if chosen is None:
            return None
        e = as_expr(e.xreplace({piece: chosen}))
    return e


def _has_integral(e: Basic) -> bool:
    from sympy.integrals.integrals import Integral
    return bool(e.atoms(Integral))


def _is_zero(e: Expr) -> bool:
    """Whether ``e`` is zero: ``cancel``, then ``simplify`` within a
    share of the time."""
    reduced = as_expr(cancel(e))
    if reduced == 0:
        return True
    simplified = attempt(lambda: simplify(reduced), _share_of_time())
    return simplified is not None and simplified == 0


def _first_integral_from_factor(field: _Field, R: Expr) -> Optional[Expr]:
    """``I`` with ``I_x = -R P`` and ``I_y = R Q`` by two quadratures,
    checked by ``D I = 0``."""
    x, y = field.x, field.y
    P, Q = field.P.as_expr(), field.Q.as_expr()
    U = _integrate(as_expr(-R*P), x, x, y)
    if U is None:
        return None
    V = as_expr(R*Q - U.diff(y))
    V = as_expr(cancel(V))
    if V.has(x):
        simplified = attempt(lambda: simplify(V), _share_of_time())
        if simplified is None or simplified.has(x):
            return None
        V = as_expr(simplified)
    W = _integrate(V, y, x, y) if V != 0 else S.Zero
    if W is None:
        return None
    I = as_expr(U + W)
    if not _is_zero(as_expr(Q*I.diff(x) + P*I.diff(y))):
        return None
    return I


def _is_constant_integral(I: Expr, x: Symbol, y: Symbol) -> bool:
    return not (I.has(x) or I.has(y))


def darboux_first_integral(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, degree: int = 4,
                           extension: Optional[Expr] = None) -> Optional[Expr]:
    """A first integral ``I(x, y)`` of ``y' = P/Q`` (``Q I_x + P I_y = 0``)
    by the Prelle–Singer procedure with Darboux polynomials of degree at
    most ``degree``: ``prod f_i**c_i exp(...)`` when a combination of the
    cofactors vanishes, else the quadrature of the Darboux integrating
    factor; ``None`` when neither is found.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.darboux import darboux_first_integral
    >>> darboux_first_integral(y*(x - 1), x*(1 - y), x, y)      # Lotka–Volterra
    -x - y + log(x) + log(y)
    >>> darboux_first_integral(x + y, x - y, x, y)
    -log(x**2 + y**2)/2 - atan(x/y)
    >>> darboux_first_integral(x*y**2 - y, x**2*y - x, x, y, degree=2)
    x/y
    """
    field = _field_of(P, Q, x, y, extension)
    return _first_integral(field, degree)


def _first_integral(field: _Field, degree: int) -> Optional[Expr]:
    """The first integral with the Darboux polynomials of the lowest
    degree which gives one (the Prelle–Singer loop: the bound is raised
    one by one up to ``degree``); over the rationals, when nothing is
    found and ``x P_d - y Q_d`` has irreducible quadratic factors, the
    search is repeated over the field of their roots (the conjugate
    lines with their own exponents)."""
    integral = _first_integral_over(field, degree)
    if integral is not None or not field.K.is_QQ:
        return integral
    extension = _quadratic_extension(field)
    if extension is None:
        return None
    extended = _field_of(field.P.as_expr(), field.Q.as_expr(), field.x, field.y, extension)
    return _first_integral_over(extended, degree)


def _first_integral_over(field: _Field, degree: int) -> Optional[Expr]:
    polynomials: dict[Expr, tuple[Poly, Poly]] = {}
    for N in range(1, degree + 1):
        _add_darboux(field, N, polynomials)
        darboux = _Darboux(field, list(polynomials.values()), N)
        direct = darboux.first_integral()
        if direct is not None:
            return _tidy(direct)
        R = darboux.integrating_factor()
        if R is None:
            continue
        integral = _first_integral_from_factor(field, R)
        if integral is not None and not _is_constant_integral(integral, field.x, field.y):
            return _tidy(integral)
    return None


def _quadratic_extension(field: _Field) -> Optional[Expr]:
    """The square root of the discriminant of an irreducible quadratic
    factor of ``x P_d - y Q_d`` (the first one), or ``None``."""
    from sympy.functions.elementary.miscellaneous import sqrt
    _, _, H = _top_parts(field)
    if H.is_zero:
        return None
    x, y = field.x, field.y
    for h, _ in H.factor_list()[1]:
        if _total_degree(h) != 2:
            continue
        quadratic = Poly(h.as_expr().subs(y, 1), x)
        if quadratic.degree() != 2:
            quadratic = Poly(h.as_expr().subs(x, 1), y)
        discriminant = as_expr(quadratic.discriminant())
        if discriminant.is_Rational and discriminant >= 0:
            # reducible, or a double line: nothing new
            continue
        return as_expr(sqrt(discriminant))
    return None


def _add_darboux(field: _Field, N: int, found: dict[Expr, tuple[Poly, Poly]]) -> None:
    """The irreducible Darboux polynomials of degree ``N`` added to
    ``found`` (keyed by their expression, monic)."""
    for f, _ in _darboux_of_degree(field, N):
        for h in _irreducible_factors(field, f):
            key = as_expr(h.as_expr())
            if key in found:
                continue
            cofactor = field.cofactor_of(h)
            if cofactor is not None:
                found[key] = (h, cofactor)


def _tidy(I: Expr) -> Expr:
    """A first integral in a readable form: cancelled as a rational
    function of its atoms when that is shorter (``-exp(x**2)/(2 y**2) -
    sqrt(pi) erfi(x)/2`` for the quadrature ``-(sqrt(pi) y**3 erfi(x)/2 +
    y exp(x**2)/2)/y**3``)."""
    return min([I, as_expr(cancel(I))], key=lambda e: int(count_ops(e)))


def prelle_singer(equation: Basic, f: AppliedUndef, degree: int = 4, extension: Optional[Expr] = None) -> Optional[Eq]:
    """The implicit solution ``Eq(I(x, y), C1)`` of a first order equation
    ``y' = P(x, y)/Q(x, y)`` with polynomial ``P`` and ``Q`` from a first
    integral found by the Prelle–Singer procedure (Darboux polynomials of
    degree at most ``degree``), or ``None``. Every step runs under the
    time limit of the settings.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.darboux import prelle_singer
    >>> y = Function('y')(x)
    >>> prelle_singer(y.diff(x) - (x + y)/(x - y), y)
    Eq(-log(x**2 + y(x)**2)/2 - atan(x/y(x)), C1)
    >>> prelle_singer(x*(1 - y)*y.diff(x) - y*(x - 1), y)
    Eq(-x - y(x) + log(x) + log(y(x)), C1)
    >>> prelle_singer(x**2*y.diff(x) - x*y + 1, y)
    Eq(y(x), C1*x + 1/(2*x))
    """
    parsed = vector_field(equation, f)
    if parsed is None:
        return None
    P, Q, x, y = parsed
    try:
        field = _field_of(P, Q, x, y, extension)
    except ValueError:
        return None
    integral = attempt(lambda: _first_integral(field, degree), settings.timeout)
    if integral is None:
        return None
    C1 = Symbol('C1')
    implicit = Eq(integral.xreplace({y: f}), C1)
    explicit = _explicit(integral, x, y, C1)
    if explicit is None:
        return implicit
    return Eq(f, explicit.xreplace({y: f}))


def _explicit(integral: Expr, x: Symbol, y: Symbol, C1: Symbol) -> Optional[Expr]:
    """``y`` from ``I(x, y) = C1`` when ``I`` is a rational function of
    ``y`` and the equation has one root in ``y`` (``C1 x + 1/(2 x)`` for
    ``(2 x y - 1)/(2 x**2) = C1``), else ``None``: a transcendental
    relation is left implicit rather than cut to one branch (``-LambertW(
    -exp(C1 + x)/x)`` for the Lotka–Volterra integral)."""
    from sympy.solvers.solvers import solve
    if not integral.is_rational_function(y):
        return None
    solutions = attempt(lambda: solve(Eq(integral, C1), y), _share_of_time())
    if not isinstance(solutions, list) or len(solutions) != 1:
        return None
    solution = solutions[0]
    if not isinstance(solution, Expr) or solution.has(y):
        return None
    return solution
