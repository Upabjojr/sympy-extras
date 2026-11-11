"""Janet bases of linear systems of partial differential equations.

A linear system `\\sum_{a, J} c_{a J}(x)\\, \\partial^J u_a = f(x)` with
coefficients rational in the independent variables generates a left
module over the ring of differential operators. A *Janet basis* is, for
that module, what a Gröbner basis is for a polynomial ideal: a canonical
system of generators which contains all the integrability conditions, so
that

* a linear differential consequence of the system reduces to zero, and
  only such an expression does (:meth:`JanetBasis.reduce`,
  :meth:`JanetBasis.contains`);
* the system is inconsistent exactly when the basis is `\\{1\\}`;
* two systems have the same solutions (the same differential consequences)
  exactly when their Janet bases for the same ranking are equal;
* the derivatives which are not derivatives of a leader, the *parametric*
  derivatives, are the Taylor coefficients of the solutions which can be
  chosen freely (Riquier's existence theorem), so that the size of the
  solution space is read off the leaders: its dimension when it is
  finite, its Hilbert series and polynomial otherwise;
* with an elimination ranking the equations of the basis which contain
  only the lower functions generate all the consequences of the system
  for those functions (compatibility conditions of right-hand sides,
  elimination of unknowns).

The algorithm is the one of Janet, in the formulation of [Robertz]_ and
[Gerdt]_. For a finite set `U` of multi-indices, the variable `x_i` is
*multiplicative* for `u \\in U` when

.. math:: u_i = \\max \\{ v_i : v \\in U,\\ v_1 = u_1, \\ldots, v_{i-1} = u_{i-1} \\} .

The *involutive cone* of `u` consists of the derivatives of `u` by its
multiplicative variables only; the cones of the elements of `U` are
disjoint, and `U` is *complete* when they cover all the derivatives of
its elements. The algorithm

1. autoreduces the system (no term of an equation is a derivative of the
   leader of another equation);
2. completes the set of the leaders, by adjoining derivatives of the
   equations by non multiplicative variables;
3. reduces the derivative of every equation by each of its non
   multiplicative variables, using for a term only the equation whose
   cone contains it (the *Janet normal form*). The remainders which are
   not zero are the integrability conditions: they are adjoined and the
   algorithm starts again. When all the remainders vanish the system is
   *passive* (involutive) and is the Janet basis.

Every equation is divided by the coefficient of its leader, so that the
basis describes the solutions at the generic points: the points where a
denominator of the basis vanishes are singular and excluded.

The complement of the leaders decomposes into disjoint cones as well
(Janet's decomposition), which gives the Hilbert series
`\\sum t^{|v|} / (1 - t)^{k}` over the cones with vertex `v` and `k`
multiplicative variables. For an orderly ranking its coefficient of `t^q`
is the number of the Taylor coefficients of order `q` which can be chosen
freely.

References
==========

.. [Janet] M. Janet, Leçons sur les systèmes d'équations aux dérivées
   partielles, Gauthier-Villars 1929.
.. [Robertz] D. Robertz, Formal Algorithmic Elimination for PDEs, Lecture
   Notes in Mathematics 2121, Springer 2014, section 2.1.
.. [Gerdt] V. P. Gerdt, Yu. A. Blinkov, Involutive bases of polynomial
   ideals, Math. Comput. Simulation 45 (1998), 519-541.
.. [Schwarz] F. Schwarz, Algorithmic Lie Theory for Solving Ordinary
   Differential Equations, Chapman & Hall 2008, chapter 2.
.. [Seiler] W. M. Seiler, Involution, Springer 2010.

The module works on objects of the core and imports no SymPy; the forms
have coefficients in ``ZZ[p, x]`` (fraction-free).
:mod:`sympy_extras.polys.differential_sympy` is the face on SymPy
objects.
"""
from __future__ import annotations

from fractions import Fraction
from math import comb, factorial
from typing import Final, Optional, Sequence

from sympy_extras.polys.rings import Monomial
from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import AppliedUndef, expand
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, oo, number_from_fraction
from sympy_extras.core.power import pow
from sympy_extras.core.symbol import Symbol

from .ring import (Coefficient, DifferentialRing, Equation, IntegerCoefficient, Jet, integer_product,
                   integer_quotient, unit_jet)

__all__ = ['JanetBasis', 'janet_basis', 'janet_multiplicative_variables', 'LinearForm', 'Cone']

#: a linear differential polynomial: the coefficients of the derivatives
#: (the derivative of index -1 stands for the inhomogeneous part)
LinearForm = dict[Jet, IntegerCoefficient]
#: a cone of multi-indices: the vertex and the flags of the multiplicative variables
Cone = tuple[Monomial, tuple[bool, ...]]


def janet_multiplicative_variables(monomials: Sequence[Monomial]) -> list[tuple[bool, ...]]:
    """The multiplicative variables of Janet's division, for every element
    of a finite set of multi-indices.

    The variable of index ``i`` is multiplicative for ``u`` when ``u[i]``
    is the largest ``i``-th exponent among the elements which agree with
    ``u`` in the exponents before ``i``.

    Examples
    ========

    The set `\\{x_1^2 x_2, x_1 x_2^2, x_2^3\\}`:

    >>> from sympy_extras.polys.differential import janet_multiplicative_variables
    >>> janet_multiplicative_variables([(2, 1), (1, 2), (0, 3)])
    [(True, True), (False, True), (False, True)]
    """
    result: list[tuple[bool, ...]] = []
    for u in monomials:
        flags: list[bool] = []
        for i in range(len(u)):
            top = max(v[i] for v in monomials if v[:i] == u[:i])
            flags.append(u[i] == top)
        result.append(tuple(flags))
    return result


def _complement(monomials: Sequence[Monomial], n: int) -> list[Cone]:
    """Janet's decomposition of the complement of the multiples of a
    complete set of multi-indices into disjoint cones."""
    if not monomials:
        return [((0,)*n, (True,)*n)]
    cones: list[Cone] = []

    def visit(group: list[Monomial], i: int, prefix: Monomial, flags: tuple[bool, ...]) -> None:
        if i == n:
            return
        values = sorted({u[i] for u in group})
        top = values[-1]
        for d in range(top):
            if d not in values:
                cones.append((prefix + (d,) + (0,)*(n - i - 1), flags + (False,) + (True,)*(n - i - 1)))
        for d in values:
            visit([u for u in group if u[i] == d], i + 1, prefix + (d,), flags + (d == top,))

    visit(list(monomials), 0, (), ())
    return cones


def _add(form: LinearForm, jet: Jet, c: IntegerCoefficient) -> None:
    value = form[jet] + c if jet in form else c
    if value:
        form[jet] = value
    else:
        form.pop(jet, None)


class _Forms:
    """The arithmetic of linear forms over a ring."""

    def __init__(self, ring: DifferentialRing) -> None:
        self.ring: Final[DifferentialRing] = ring
        self.unit: Final[Jet] = unit_jet(ring.n)

    def parse(self, equation: Equation) -> tuple[LinearForm, Coefficient]:
        """The form of an equation with coefficients in the ring of the
        integers, and the element of the field it was multiplied by."""
        terms = self.ring.terms(equation)
        for monomial in terms:
            if monomial and not (len(monomial) == 1 and monomial[0][1] == 1):
                raise ValueError("%s is not linear in the functions and their derivatives: "
                                 "use rosenfeld_groebner" % (equation,))
        form: LinearForm = {}
        scale = self.ring.domain.one
        for monomial, c in self.ring.integral(terms).items():
            form[monomial[0][0] if monomial else self.unit] = c
            scale = self.ring.domain.exquo(self.ring.field_element(c), terms[monomial])
        return form, scale

    def to_expr(self, form: LinearForm, divisor: Optional[Coefficient] = None) -> Expr:
        """The expression of the form divided by ``divisor`` (by the
        coefficient of its leader by default)."""
        if not form:
            return Integer(0)
        ring, domain = self.ring, self.ring.domain
        below = ring.field_element(form[self.leader(form)]) if divisor is None else divisor
        ordered = sorted(form, key=ring.rank_key, reverse=True)
        return add(*[ring.coefficient_expr(domain.exquo(ring.field_element(form[jet]), below))
                     * ring.to_expr(jet) for jet in ordered])

    def leader(self, form: LinearForm) -> Jet:
        return max(form, key=self.ring.rank_key)

    def primitive(self, form: LinearForm) -> LinearForm:
        """The form divided by the greatest common divisor of its
        coefficients, with a positive coefficient of the leader."""
        Z = self.ring.integers
        one = Z.one
        values = list(form.values())
        divisor = values[0]
        for c in values[1:]:
            if divisor == one:
                break
            divisor = Z.gcd(divisor, c)
        if (form[self.leader(form)].LC < 0) != (divisor.LC < 0):
            divisor = -divisor
        if divisor == one:
            return form
        return {jet: integer_quotient(c, divisor) for jet, c in form.items()}

    def derivative(self, form: LinearForm, i: int) -> LinearForm:
        result: LinearForm = {}
        for jet, c in form.items():
            d = self.ring.integer_derivative(c, i)
            if d:
                _add(result, jet, d)
            if jet[0] >= 0:
                _add(result, self.ring.differentiate(jet, i), c)
        return result

    def prolong(self, form: LinearForm, theta: Monomial) -> LinearForm:
        for i, k in enumerate(theta):
            for _ in range(k):
                form = self.derivative(form, i)
        return form


class _Division:
    """The divisors among a list of forms: every derivative of a
    leader (the conventional division) or the derivatives in the
    involutive cones only (Janet's division)."""

    def __init__(self, forms: _Forms, elements: list[LinearForm],
                 multiplicative: Optional[list[tuple[bool, ...]]]) -> None:
        self.forms: Final[_Forms] = forms
        self.elements: Final[list[LinearForm]] = elements
        self.leaders: Final[list[Jet]] = [forms.leader(p) for p in elements]
        self.multiplicative: Final[Optional[list[tuple[bool, ...]]]] = multiplicative
        self._prolonged: Final[dict[tuple[int, Monomial], LinearForm]] = {}

    def divisor(self, jet: Jet, skip: int = -1) -> Optional[tuple[int, Monomial]]:
        """The index of an element whose leader divides the derivative,
        with the quotient."""
        a, J = jet
        for k, (b, L) in enumerate(self.leaders):
            if b != a or k == skip or any(j < l for j, l in zip(J, L)):
                continue
            if self.multiplicative is not None:
                flags = self.multiplicative[k]
                if any(j > l and not flag for j, l, flag in zip(J, L, flags)):
                    continue
            return k, tuple(j - l for j, l in zip(J, L))
        return None

    def prolonged(self, k: int, theta: Monomial) -> LinearForm:
        key = (k, theta)
        form = self._prolonged.get(key)
        if form is None:
            if not any(theta):
                form = self.elements[k]
            else:
                i = next(i for i, t in enumerate(theta) if t)
                lower = theta[:i] + (theta[i] - 1,) + theta[i + 1:]
                form = self.forms.derivative(self.prolonged(k, lower), i)
            self._prolonged[key] = form
        return form

    def normal_form(self, form: LinearForm, skip: int = -1) -> tuple[LinearForm, IntegerCoefficient]:
        """A multiple of the form with every divisible term reduced, and
        the multiplier (an element of the ring of the integers; the
        reduction is free of fractions)."""
        ring = self.forms.ring
        Z = ring.integers
        one = Z.one
        multiplier = one
        result: LinearForm = {}
        rest = dict(form)
        while rest:
            jet = max(rest, key=ring.rank_key)
            c = rest.pop(jet)
            found = self.divisor(jet, skip)
            if found is None:
                result[jet] = c
                continue
            divisor = self.prolonged(*found)
            lead = divisor[jet]
            common = Z.gcd(lead, c)
            a, b = integer_quotient(lead, common), integer_quotient(c, common)
            if a != one:
                rest = {other: integer_product(e, a) for other, e in rest.items()}
                result = {other: integer_product(e, a) for other, e in result.items()}
                multiplier = integer_product(multiplier, a)
            for other, e in divisor.items():
                if other != jet:
                    _add(rest, other, -integer_product(b, e))
        return result, multiplier


def _autoreduce(forms: _Forms, elements: Sequence[LinearForm]) -> list[LinearForm]:
    """Primitive forms none of whose terms is a derivative of the leader of
    another one."""
    current = [forms.primitive(p) for p in elements if p]
    while True:
        current.sort(key=lambda p: forms.ring.rank_key(forms.leader(p)))
        division = _Division(forms, current, None)
        for k, p in enumerate(current):
            if all(division.divisor(jet, k) is None for jet in p):
                continue
            r, _ = division.normal_form(p, skip=k)
            if r:
                current[k] = forms.primitive(r)
            else:
                del current[k]
            break
        else:
            return current


def _complete(forms: _Forms, elements: list[LinearForm]) -> tuple[list[LinearForm], list[tuple[bool, ...]]]:
    """The elements with the derivatives which make the set of the leaders
    complete for Janet's division, and the multiplicative variables."""
    ring = forms.ring
    current = list(elements)
    while True:
        multiplicative = _multiplicative(forms, current)
        division = _Division(forms, current, multiplicative)
        missing: list[tuple[tuple[int, ...], int, int]] = []
        for k, leader in enumerate(division.leaders):
            for i, flag in enumerate(multiplicative[k]):
                if not flag:
                    jet = ring.differentiate(leader, i)
                    if division.divisor(jet) is None:
                        missing.append((ring.rank_key(jet), k, i))
        if not missing:
            return current, multiplicative
        _, k, i = min(missing)
        current.append(forms.derivative(current[k], i))


def _multiplicative(forms: _Forms, elements: Sequence[LinearForm]) -> list[tuple[bool, ...]]:
    leaders = [forms.leader(p) for p in elements]
    result: list[tuple[bool, ...]] = [()]*len(leaders)
    for a in {leader[0] for leader in leaders}:
        indices = [k for k, leader in enumerate(leaders) if leader[0] == a]
        flags = janet_multiplicative_variables([leaders[k][1] for k in indices])
        for k, f in zip(indices, flags):
            result[k] = f
    return result


class JanetBasis:
    """The Janet basis of a linear system of differential equations; see
    :func:`janet_basis`.

    Attributes
    ==========

    ring : DifferentialRing
        The functions, the variables and the ranking.
    forms : list of LinearForm
        The equations, with coprime polynomial coefficients, by increasing
        leader.
    multiplicative : list of tuples of bool
        For every equation, the flags of the multiplicative variables.
    """

    def __init__(self, ring: DifferentialRing, forms: list[LinearForm],
                 multiplicative: list[tuple[bool, ...]]) -> None:
        self.ring: Final[DifferentialRing] = ring
        self.forms: Final[list[LinearForm]] = forms
        self.multiplicative: Final[list[tuple[bool, ...]]] = multiplicative
        self._arithmetic: Final[_Forms] = _Forms(ring)
        self._division: Final[_Division] = _Division(self._arithmetic, forms, multiplicative)

    def __repr__(self) -> str:
        return "JanetBasis(%s)" % (self.equations,)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JanetBasis):
            return NotImplemented
        return (self.ring.functions == other.ring.functions and self.ring.variables == other.ring.variables
                and self.ring.blocks == other.ring.blocks and self.forms == other.forms)

    def __hash__(self) -> int:
        return hash((self.ring.functions, self.ring.variables, len(self.forms)))

    # ------------------------------------------------------------------
    # the equations

    @property
    def is_consistent(self) -> bool:
        """Whether the system has solutions: the basis is not `\\{1\\}`."""
        return not any(self._division.leaders[k][0] < 0 for k in range(len(self.forms)))

    def _shown(self) -> list[int]:
        """The equations other than the constancy of a function in a
        variable it does not depend on."""
        hidden = set(self.ring.constancy())
        result: list[int] = []
        for k, leader in enumerate(self._division.leaders):
            a, J = leader
            if a >= 0 and any(j and (a, (0,)*i + (1,) + (0,)*(self.ring.n - i - 1)) in hidden
                              for i, j in enumerate(J)):
                continue
            result.append(k)
        return result

    @property
    def equations(self) -> list[Expr]:
        """The expressions which vanish, each with the coefficient one at
        its leader, by increasing leader."""
        return [self._arithmetic.to_expr(self.forms[k]) for k in self._shown()]

    @property
    def leaders(self) -> list[Expr]:
        """The leaders of the equations."""
        return [self.ring.to_expr(self._division.leaders[k]) for k in self._shown()]

    @property
    def multiplicative_variables(self) -> list[tuple[Symbol, ...]]:
        """The multiplicative variables of the equations."""
        return [tuple(s for s, flag in zip(self.ring.variables, self.multiplicative[k]) if flag)
                for k in self._shown()]

    def reduce(self, expr: Equation) -> Expr:
        """The Janet normal form of a linear differential expression: the
        unique expression in the parametric derivatives which differs from
        it by a consequence of the system.

        >>> from sympy_extras.core import Function, symbols
        >>> from sympy_extras.polys.differential import janet_basis
        >>> x, y = symbols('x y')
        >>> u = Function('u')(x, y)
        >>> J = janet_basis([u.diff(x, 2) - u, u.diff(y) - u.diff(x)], [u])
        >>> J.reduce(u.diff(x, y, y) + u.diff(y))
        2*Derivative(u(x, y), y)
        """
        form, scale = self._arithmetic.parse(expr)
        reduced, multiplier = self._division.normal_form(form)
        return self._arithmetic.to_expr(reduced, self.ring.domain.mul(scale, self.ring.field_element(multiplier)))

    def contains(self, expr: Equation) -> bool:
        """Whether the expression vanishes on the solutions of the system:
        it is a linear differential consequence of the equations."""
        return not self._division.normal_form(self._arithmetic.parse(expr)[0])[0]

    def __contains__(self, expr: Equation) -> bool:
        return self.contains(expr)

    # ------------------------------------------------------------------
    # the size of the solution space

    def _require_consistent(self) -> None:
        if not self.is_consistent:
            raise ValueError("the system is inconsistent: it has no solutions")

    def _cones(self) -> list[tuple[int, Cone]]:
        """The cones of the parametric derivatives, per function."""
        self._require_consistent()
        result: list[tuple[int, Cone]] = []
        for a in range(len(self.ring.functions)):
            leaders = [leader[1] for leader in self._division.leaders if leader[0] == a]
            for cone in _complement(leaders, self.ring.n):
                result.append((a, cone))
        return result

    def is_parametric(self, derivative: Expr) -> bool:
        """Whether the derivative is not a derivative of a leader."""
        jet = self.ring.jet(derivative)
        if jet is None:
            raise ValueError("%s is not a derivative of a function of the ring" % (derivative,))
        return self._division.divisor(jet) is None

    def parametric_derivatives(self, order: int) -> list[Expr]:
        """The parametric derivatives up to the given order, by increasing
        rank: the Taylor coefficients of a solution which can be chosen
        freely."""
        return [self.ring.to_expr(jet) for jet in self._parametric(order)]

    def _parametric(self, order: int) -> list[Jet]:
        self._require_consistent()
        jets: list[Jet] = []
        for a in range(len(self.ring.functions)):
            for J in _multi_indices(self.ring.n, order):
                if self._division.divisor((a, J)) is None:
                    jets.append((a, J))
        return sorted(jets, key=self.ring.rank_key)

    def hilbert_function(self, order: int) -> int:
        """The number of the parametric derivatives of exactly the given
        order."""
        total = 0
        for _, (vertex, flags) in self._cones():
            k = sum(flags)
            d = order - sum(vertex)
            if d < 0:
                continue
            if k == 0:
                total += 1 if d == 0 else 0
            else:
                total += comb(d + k - 1, k - 1)
        return total

    def hilbert_series(self, t: Symbol) -> Expr:
        """The generating function of :meth:`hilbert_function`."""
        return add(*[t**sum(vertex)/(1 - t)**sum(flags) for _, (vertex, flags) in self._cones()])

    def hilbert_polynomial(self, s: Symbol) -> Expr:
        """The polynomial which :meth:`hilbert_function` equals for the
        large orders `s`; zero when the solution space has a finite
        dimension. Its degree `d` and its leading coefficient `c/d!` say
        that the general solution depends on `c` arbitrary functions of
        `d + 1` variables."""
        # binomial(s - |v| + k - 1, k - 1) as a polynomial in s, summed
        total: list[Fraction] = []
        for _, (vertex, flags) in self._cones():
            k = sum(flags)
            if k:
                shift = k - 1 - sum(vertex)
                coefficients = [Fraction(1)]
                for i in range(k - 1):
                    # times (s + shift - i)/(i + 1)
                    c = Fraction(shift - i, i + 1)
                    coefficients = [(coefficients[j] if j < len(coefficients) else Fraction(0))*c
                                    + (coefficients[j - 1]/(i + 1) if j > 0 else Fraction(0))
                                    for j in range(len(coefficients) + 1)]
                for j, c in enumerate(coefficients):
                    while len(total) <= j:
                        total.append(Fraction(0))
                    total[j] += c
        return add(*[mul(number_from_fraction(c), pow(s, Integer(j))) for j, c in enumerate(total) if c])

    @property
    def dimension(self) -> Expr:
        """The dimension of the solution space (of the affine space of the
        solutions for an inhomogeneous system): the number of the
        parametric derivatives, ``oo`` when there are infinitely many."""
        cones = self._cones()
        if any(any(flags) for _, (_, flags) in cones):
            return oo
        return Integer(len(cones))

    # ------------------------------------------------------------------
    # power series

    def series_solution(self, order: int, point: Optional[Sequence[ExprLike]] = None,
                        constant: str = 'C') -> dict[AppliedUndef, Expr]:
        """The Taylor polynomials of degree ``order`` of the general
        solution at a point (the origin by default).

        The values of the parametric derivatives at the point are the
        arbitrary constants ``C0, C1, ...`` (numbered by increasing rank);
        the other Taylor coefficients follow from the Janet normal forms
        of the derivatives. The point must not be a zero of a denominator
        of the basis.

        >>> from sympy_extras.core import Function, symbols
        >>> from sympy_extras.polys.differential import janet_basis
        >>> x = symbols('x')
        >>> u = Function('u')(x)
        >>> janet_basis([u.diff(x, 2) + u], [u]).series_solution(4)[u]
        C0*x**4/24 - C0*x**2/2 + C0 - C1*x**3/6 + C1*x
        """
        self._require_consistent()
        ring = self.ring
        n = ring.n
        values: list[Expr] = [to_expr(v) for v in point] if point is not None else [Integer(0)]*n
        if len(values) != n:
            raise ValueError("the point must have %d coordinates" % n)
        at: dict[Symbol, Expr] = dict(zip(ring.variables, values))
        for form in self.forms:
            lead = ring.integer_expr(form[self._arithmetic.leader(form)], at)
            if lead == 0:
                raise ValueError("the point %s is singular: the coefficient of a leader vanishes there"
                                 % (tuple(values),))
        constants: dict[Jet, Symbol] = {}
        expansions: list[tuple[int, Monomial, LinearForm, IntegerCoefficient]] = []
        needed: set[Jet] = set()
        for a in range(len(self.ring.functions)):
            for J in _multi_indices(n, order):
                form, multiplier = self._division.normal_form({(a, J): self.ring.integers.one})
                expansions.append((a, J, form, multiplier))
                needed |= {jet for jet in form if jet[0] >= 0}
        for k, jet in enumerate(sorted(needed, key=self.ring.rank_key)):
            constants[jet] = Symbol('%s%d' % (constant, k))
        result: dict[AppliedUndef, Expr] = {f: Integer(0) for f in ring.functions}
        for a, J, form, multiplier in expansions:
            value: Expr = Integer(0)
            below = ring.integer_expr(multiplier, at)
            for jet, c in form.items():
                factor = ring.integer_expr(c, at)/below
                value += factor*(constants[jet] if jet[0] >= 0 else Integer(1))
            term = value
            for s, x0, j in zip(ring.variables, values, J):
                term = term*(s - x0)**j/factorial(j)
            result[ring.functions[a]] += term
        return {f: expand(e) for f, e in result.items()}


def _multi_indices(n: int, order: int) -> list[Monomial]:
    """The multi-indices of total order at most ``order``."""
    result: list[Monomial] = [()]
    for _ in range(n):
        result = [J + (j,) for J in result for j in range(order - sum(J) + 1)]
    return result


def janet_basis(equations: Sequence[Equation], functions: Sequence[Basic],
                variables: Optional[Sequence[Basic]] = None,
                ranking: Optional[Sequence[Sequence[Basic]]] = None) -> JanetBasis:
    """The Janet basis of a linear system of differential equations with
    coefficients rational in the variables.

    Parameters
    ==========

    equations : sequence of Expr or Eq
        Expressions which vanish, or equalities, linear in the functions
        and their derivatives; a term without a function is a right-hand
        side. The symbols which are not variables are constant parameters
        (and are taken to be generic: a coefficient which vanishes only
        for special values of them is not zero).
    functions : sequence of applied functions
        The unknown functions, such as ``u(x, y)``.
    variables : sequence of Symbol, optional
        The independent variables, in the order used by the ranking and by
        Janet's division (the arguments of the functions by default).
    ranking : sequence of sequences of functions, optional
        The blocks of an elimination ranking, the functions to eliminate
        first; an orderly ranking by default. See
        :class:`~sympy_extras.polys.differential.ring.DifferentialRing`.

    Returns
    =======

    JanetBasis

    Examples
    ========

    Janet's example: the system `u_{zz} + y\\,u_{xx} = 0`, `u_{yy} = 0` has
    hidden integrability conditions of order three and four, and a solution
    space of dimension 12 [Janet]_:

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.polys.differential import janet_basis
    >>> x, y, z = symbols('x y z')
    >>> u = Function('u')(x, y, z)
    >>> J = janet_basis([u.diff(z, 2) + y*u.diff(x, 2), u.diff(y, 2)], [u])
    >>> J.equations[:2]
    [Derivative(u(x, y, z), (y, 2)), Derivative(u(x, y, z), (x, 2)) + Derivative(u(x, y, z), (z, 2))/y]
    >>> len(J.equations)
    7
    >>> J.dimension
    12
    >>> t = symbols('t')
    >>> J.hilbert_series(t)
    t**4 + 3*t**3 + 4*t**2 + 3*t + 1

    An inconsistent system:

    >>> v = Function('v')(x, y)
    >>> janet_basis([v.diff(x) - y, v.diff(y)], [v]).equations
    [1]

    The compatibility condition of `u_x = f`, `u_y = g`, by the
    elimination of `u`:

    >>> w, f, g = Function('w')(x, y), Function('f')(x, y), Function('g')(x, y)
    >>> J = janet_basis([w.diff(x) - f, w.diff(y) - g], [w, f, g], ranking=[[w], [f, g]])
    >>> J.equations[0]
    Derivative(f(x, y), y) - Derivative(g(x, y), x)
    """
    given = list(equations)
    chosen = list(variables) if variables is not None else None
    probe = DifferentialRing(functions, chosen, ranking)
    symbols: set[Symbol] = set()
    for equation in given:
        symbols |= equation.free_symbols
    parameters = sorted(symbols - set(probe.variables), key=lambda s: s.name)
    ring = DifferentialRing(functions, chosen, ranking, parameters)
    forms = _Forms(ring)
    one = ring.integers.one
    elements = [forms.parse(equation)[0] for equation in given]
    elements += [{jet: one} for jet in ring.constancy()]
    current = _autoreduce(forms, elements)
    while True:
        inconsistent = [p for p in current if forms.leader(p)[0] < 0]
        if inconsistent:
            return JanetBasis(ring, [{forms.unit: one}], [(True,)*ring.n])
        completed, multiplicative = _complete(forms, current)
        division = _Division(forms, completed, multiplicative)
        conditions: list[LinearForm] = []
        for k in range(len(completed)):
            for i, flag in enumerate(multiplicative[k]):
                if not flag:
                    r, _ = division.normal_form(forms.derivative(completed[k], i))
                    if r:
                        conditions.append(forms.primitive(r))
        if not conditions:
            break
        current = _autoreduce(forms, completed + conditions)
    reduced: list[LinearForm] = []
    for k, p in enumerate(completed):
        leader = division.leaders[k]
        tail, multiplier = division.normal_form({jet: c for jet, c in p.items() if jet != leader})
        tail[leader] = integer_product(p[leader], multiplier)
        reduced.append(forms.primitive(tail))
    order = sorted(range(len(reduced)), key=lambda k: ring.rank_key(division.leaders[k]))
    return JanetBasis(ring, [reduced[k] for k in order], [multiplicative[k] for k in order])
