r"""Real solutions of zero-dimensional systems counted without solving them:
Hermite's quadratic form and sign determination.

Let $I \subset \mathbb{Q}[x_1, \ldots, x_n]$ be a zero-dimensional ideal,
$A = \mathbb{Q}[x]/I$ its quotient algebra (a vector space of finite
dimension $D$, with the standard monomials $b_1, \ldots, b_D$ of a Gröbner
basis as basis) and $Z$ the finite set of its complex zeros. For a
polynomial $h$, the trace $\operatorname{Tr}(h)$ of the multiplication by
$h$ on $A$ is $\sum_{z \in Z} \mu(z) h(z)$, $\mu(z)$ the multiplicity. The
*Hermite quadratic form* of a polynomial $g$ is

.. math::

    H_g(p, q) = \operatorname{Tr}(g p q), \qquad p, q \in A,

and Hermite's theorem, in the form of Pedersen, Roy and Szpirglas, says
that

* its **signature** is the *Tarski query*
  $\operatorname{TaQ}(g) = \#\{z \in Z \cap \mathbb{R}^n : g(z) > 0\} -
  \#\{z \in Z \cap \mathbb{R}^n : g(z) < 0\}$;
* its **rank** is the number of distinct complex zeros at which
  $g \ne 0$.

The multiplicities do not enter: the real zeros are counted once each.
With $g = 1$ the signature is the number of distinct real zeros, and the
queries of $1, g, g^2$ give the numbers $c_0, c_+, c_-$ of real zeros at
which $g$ is zero, positive and negative through the invertible system

.. math::

    \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & -1 \\ 0 & 1 & 1 \end{pmatrix}
    \begin{pmatrix} c_0 \\ c_+ \\ c_- \end{pmatrix} =
    \begin{pmatrix} \operatorname{TaQ}(1) \\ \operatorname{TaQ}(g) \\
    \operatorname{TaQ}(g^2) \end{pmatrix}.

For several polynomials $P_1, \ldots, P_s$ the counts of the $3^s$ sign
conditions are found by the sign determination scheme of Ben-Or, Kozen and
Reif as presented by Basu, Pollack and Roy (Algorithm 10.11): the
polynomials are added one at a time, the system above is tensored with the
one of the realizable sign conditions found so far, the conditions of
count zero are dropped, and the products of powers of the $P_i$ whose rows
stay independent on the remaining conditions are kept, so that at every
step at most $3 r$ Tarski queries are computed, $r \le \# (Z \cap
\mathbb{R}^n)$ the number of realizable conditions.

Everything is exact linear algebra over $\mathbb{Q}$: the matrix of
$H_g$ has the entries $\operatorname{Tr}(g\, b_i b_j)$, which are the
linear form $\ell_g(h) = \operatorname{Tr}(g h)$ applied to the normal
forms of the products $b_i b_j$ (computed once per ideal); $\ell_1$ is the
vector of traces $\operatorname{Tr}(b_k)$ and $\ell_{g h} = M_g^T \ell_h$,
$M_g$ the multiplication matrix of $g$. The signature is computed by
Lagrange's reduction of the symmetric matrix to a diagonal one by
congruence (Sylvester's law of inertia), with a $2 \times 2$ step when the
remaining diagonal is zero; no floating point number is involved.

References
==========

.. [PRS93] P. Pedersen, M.-F. Roy, A. Szpirglas, Counting real zeros in
   the multivariate case, in Computational Algebraic Geometry, Progress in
   Mathematics 109, Birkhäuser (1993), 203–224.
.. [BPR] S. Basu, R. Pollack, M.-F. Roy, Algorithms in Real Algebraic
   Geometry, 2nd ed., Springer (2006): Theorem 4.102 (Hermite's quadratic
   form), Section 10.3 and Algorithm 10.11 (sign determination).
.. [BKR86] M. Ben-Or, D. Kozen, J. Reif, The complexity of elementary
   algebra and geometry, Journal of Computer and System Sciences 32
   (1986), 251–264.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence, Union

from sympy.core.expr import Expr
from sympy.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import And, Boolean, BooleanFalse, BooleanTrue
from sympy.matrices.dense import MutableDenseMatrix
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import CoercionFailed, PolynomialError
from sympy.polys.polytools import Poly
from sympy.polys.rings import PolyElement, PolyRing

from sympy_extras._typing import DomainElement, Monomial, Sign, Truth, as_expr

from .ideals import Ideal

__all__ = ['SignCounts', 'hermite_matrix', 'tarski_query', 'count_real_solutions',
           'count_complex_solutions', 'real_sign_counts', 'sign_determination',
           'decide_zero_dimensional']

#: a system: an ideal, or the polynomials which generate it
System = Union[Ideal, Sequence[Union[Expr, Poly]]]
#: a vector of rationals: coordinates in the basis of standard monomials,
#: or a linear form on the quotient algebra
Vector = list[DomainElement]
#: a square matrix of rationals, as a list of rows
Rows = list[list[DomainElement]]
#: a sign condition on a list of polynomials
SignCondition = tuple[Sign, ...]


class SignCounts(NamedTuple):
    """The numbers of distinct real zeros at which a polynomial is zero,
    positive and negative."""
    zero: int
    positive: int
    negative: int


class _Algebra:
    """The quotient algebra of a zero-dimensional ideal over ``QQ``, with
    the data every Hermite form needs: the normal forms of the products of
    two standard monomials and the traces of the standard monomials."""

    def __init__(self, ideal: Ideal) -> None:
        if ideal.domain != QQ:
            raise NotImplementedError("real zeros are counted for ideals over QQ, not %s" % ideal.domain)
        self.ideal: Ideal = ideal
        self.empty: bool = ideal.is_whole_ring()
        self.ring: PolyRing = ideal.ring()
        self.basis: list[PolyElement] = []
        self.monomials: list[Monomial] = []
        self.index: dict[Monomial, int] = {}
        self.products: list[list[Vector]] = []
        self.trace: Vector = []
        if self.empty:
            return
        ideal._require_zero_dimensional()
        self.monomials = [self.ring.from_expr(b).LM for b in ideal.standard_monomials()]
        self.index = {m: i for i, m in enumerate(self.monomials)}
        one = self.ring.domain.one
        self.basis = [self.ring.from_dict({m: one}) for m in self.monomials]
        D = len(self.basis)
        rows: list[list[Vector]] = [[[] for _ in range(D)] for _ in range(D)]
        for i in range(D):
            for j in range(i, D):
                v = self.coordinates(self.basis[i]*self.basis[j])
                rows[i][j] = v
                rows[j][i] = v
        self.products = rows
        self.trace = [sum((rows[k][l][l] for l in range(D)), QQ.zero) for k in range(D)]

    def coordinates(self, f: PolyElement) -> Vector:
        """The coordinates of the normal form of ``f``."""
        r = f.rem(self.ideal._basis())
        v: Vector = [QQ.zero]*len(self.monomials)
        for m, c in r.terms():
            v[self.index[m]] = c
        return v

    def element(self, g: Union[Expr, Poly, int]) -> PolyElement:
        if isinstance(g, int):
            return self.ring.from_dict({self.monomials[0]: QQ(g)}) if g else self.ring.zero
        return self.ideal._from_expr(g)

    def transposed_multiplication(self, g: PolyElement) -> Rows:
        """The transpose of the matrix of multiplication by ``g``: its row
        ``j`` holds the coordinates of the normal form of ``g b_j``."""
        return [self.coordinates(g*b) for b in self.basis]

    def compose(self, form: Vector, transposed: Rows) -> Vector:
        """The linear form ``h -> form(g h)``, for ``transposed`` the
        transposed multiplication matrix of ``g``."""
        return [sum((c*w for c, w in zip(row, form)), QQ.zero) for row in transposed]

    def hermite(self, form: Vector) -> Rows:
        """The matrix of ``(p, q) -> form(p q)`` in the standard monomials."""
        D = len(self.basis)
        H: Rows = [[QQ.zero]*D for _ in range(D)]
        for i in range(D):
            for j in range(i, D):
                v = sum((c*w for c, w in zip(self.products[i][j], form)), QQ.zero)
                H[i][j] = v
                H[j][i] = v
        return H

    def form(self, g: Union[Expr, Poly, int] = 1) -> Vector:
        """The linear form ``h -> Tr(g h)``."""
        if isinstance(g, int) and g == 1:
            return list(self.trace)
        return self.compose(self.trace, self.transposed_multiplication(self.element(g)))


def _inertia(matrix: Rows) -> tuple[int, int]:
    """The numbers of positive and negative squares of a symmetric
    rational matrix (Lagrange's reduction by congruence)."""
    A = [list(row) for row in matrix]
    active = list(range(len(A)))
    positive = negative = 0
    while active:
        pivot = next((i for i in active if A[i][i]), None)
        if pivot is None:
            pair = next(((i, j) for i in active for j in active if i < j and A[i][j]), None)
            if pair is None:
                break
            i, j = pair
            # replace e_i by e_i + e_j: the new diagonal entry is 2 A[i][j]
            for k in active:
                A[i][k] += A[j][k]
            for k in active:
                A[k][i] += A[k][j]
            pivot = i
        d = A[pivot][pivot]
        if d > 0:
            positive += 1
        else:
            negative += 1
        active.remove(pivot)
        for i in active:
            f = A[i][pivot]/d
            if f:
                for j in active:
                    A[i][j] -= f*A[pivot][j]
    return positive, negative


def _algebra(system: System, gens: Sequence[Symbol]) -> _Algebra:
    ideal = system if isinstance(system, Ideal) else Ideal(list(system), *gens)
    return _Algebra(ideal)


def hermite_matrix(system: System, g: Union[Expr, Poly, int] = 1, *gens: Symbol) -> MutableDenseMatrix:
    r"""The matrix of Hermite's quadratic form $(p, q) \mapsto
    \operatorname{Tr}(g p q)$ on the quotient algebra of a zero-dimensional
    ideal, in the basis of its standard monomials.

    Parameters
    ==========

    system : Ideal or list of Expr or Poly
        The ideal, or its generators (with rational coefficients).
    g : Expr or Poly
        The polynomial of the form (``1`` by default).
    gens : Symbols
        The variables, when ``system`` is a list of polynomials (all its
        free symbols by default).

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy_extras.polys.hermite import hermite_matrix
    >>> hermite_matrix([x**2 - 2], 1, x)
    Matrix([[2, 0], [0, 4]])
    >>> hermite_matrix([x**2 - 2], x, x)
    Matrix([[0, 4], [4, 0]])

    The first matrix is positive definite (two real roots), the second has
    signature zero (one root of each sign).
    """
    A = _algebra(system, gens)
    form = A.form(g)
    D = len(A.basis)
    H = A.hermite(form)
    return MutableDenseMatrix(D, D, [QQ.to_sympy(c) for row in H for c in row])


def tarski_query(system: System, g: Union[Expr, Poly, int] = 1, *gens: Symbol) -> int:
    r"""The Tarski query of ``g`` on the real zeros of a zero-dimensional
    system: the number of distinct real zeros at which ``g > 0`` minus the
    number at which ``g < 0``, as the signature of Hermite's quadratic
    form (Pedersen–Roy–Szpirglas).

    Parameters
    ==========

    system : Ideal or list of Expr or Poly
        The ideal, or its generators (with rational coefficients); it must
        be zero-dimensional or the whole ring.
    g : Expr or Poly
        A polynomial with rational coefficients (``1`` by default, which
        counts the real zeros).
    gens : Symbols
        The variables, when ``system`` is a list of polynomials.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.hermite import tarski_query
    >>> tarski_query([x**2 + y**2 - 1, x - 2*y], x, x, y)
    0
    >>> tarski_query([x**3 - x, y - x**2], 2*y - 1, x, y)
    1
    """
    A = _algebra(system, gens)
    if A.empty:
        return 0
    positive, negative = _inertia(A.hermite(A.form(g)))
    return positive - negative


def count_real_solutions(system: System, *gens: Symbol) -> int:
    """The number of distinct real solutions of a zero-dimensional system
    of polynomial equations with rational coefficients, found without
    solving it: the signature of Hermite's quadratic form.

    Parameters
    ==========

    system : Ideal or list of Expr or Poly
        The ideal, or the polynomials (equated to zero).
    gens : Symbols
        The variables, when ``system`` is a list of polynomials (all its
        free symbols by default).

    Raises
    ======

    NotImplementedError
        If the system has infinitely many complex solutions or its
        coefficients are not rational.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.hermite import count_real_solutions
    >>> count_real_solutions([x**2 + y**2 - 1, x - y], x, y)
    2
    >>> count_real_solutions([x**2 + 1, y], x, y)
    0

    A multiple solution is counted once:

    >>> count_real_solutions([(x - 1)**3*(x + 2), y**2], x, y)
    2
    """
    return tarski_query(system, 1, *gens)


def count_complex_solutions(system: System, *gens: Symbol) -> int:
    """The number of distinct complex solutions of a zero-dimensional
    system: the rank of Hermite's quadratic form.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.hermite import count_complex_solutions
    >>> count_complex_solutions([x**2 + 1, (y - 1)**2], x, y)
    2
    """
    A = _algebra(system, gens)
    if A.empty:
        return 0
    positive, negative = _inertia(A.hermite(A.form()))
    return positive + negative


def real_sign_counts(system: System, g: Union[Expr, Poly], *gens: Symbol) -> SignCounts:
    r"""The numbers of distinct real solutions of a zero-dimensional system
    at which ``g`` is zero, positive and negative, from the Tarski queries
    of $1$, $g$ and $g^2$.

    Parameters
    ==========

    system : Ideal or list of Expr or Poly
        The ideal, or its generators (with rational coefficients).
    g : Expr or Poly
        A polynomial with rational coefficients.
    gens : Symbols
        The variables, when ``system`` is a list of polynomials.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.hermite import real_sign_counts
    >>> real_sign_counts([x**3 - x, y - x**2], x - y, x, y)
    SignCounts(zero=2, positive=0, negative=1)
    """
    counts = sign_determination(system, [g], *gens)
    return SignCounts(counts.get((0,), 0), counts.get((1,), 0), counts.get((-1,), 0))


def _solve(matrix: Rows, rhs: Vector) -> Vector:
    """The solution of a square invertible rational system."""
    n = len(matrix)
    A = [list(row) + [b] for row, b in zip(matrix, rhs)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if A[r][col])
        A[col], A[pivot] = A[pivot], A[col]
        p = A[col][col]
        for r in range(n):
            if r != col and A[r][col]:
                f = A[r][col]/p
                A[r] = [a - f*b for a, b in zip(A[r], A[col])]
    return [A[i][n]/A[i][i] for i in range(n)]


def _independent_rows(matrix: Rows) -> list[int]:
    """The indices of the rows kept by a greedy choice of linearly
    independent rows, in order."""
    reduced: list[tuple[int, Vector]] = []  # (pivot column, row)
    chosen: list[int] = []
    for k, row in enumerate(matrix):
        v = list(row)
        for col, r in reduced:
            if v[col]:
                f = v[col]/r[col]
                v = [a - f*b for a, b in zip(v, r)]
        lead = next((c for c, a in enumerate(v) if a), None)
        if lead is not None:
            reduced.append((lead, v))
            chosen.append(k)
    return chosen


def sign_determination(system: System, polys: Sequence[Union[Expr, Poly]],
                       *gens: Symbol) -> dict[SignCondition, int]:
    r"""The realizable sign conditions of a list of polynomials on the real
    solutions of a zero-dimensional system, with the number of distinct
    real solutions realizing each (Ben-Or–Kozen–Reif sign determination on
    Tarski queries computed by Hermite's quadratic form).

    Parameters
    ==========

    system : Ideal or list of Expr or Poly
        The ideal, or its generators (with rational coefficients).
    polys : list of Expr or Poly
        The polynomials $P_1, \ldots, P_s$, with rational coefficients.
    gens : Symbols
        The variables, when ``system`` is a list of polynomials.

    Returns
    =======

    dict
        A tuple of signs (``-1``, ``0`` or ``1``) of $P_1, \ldots, P_s$ to
        the number of real solutions at which the polynomials have these
        signs; the conditions realized by no solution are absent.

    Examples
    ========

    The four points $(\pm 1, \pm 1)$ with the signs of $x + y$ and $x$:

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.hermite import sign_determination
    >>> sign_determination([x**2 - 1, y**2 - 1], [x + y, x], x, y)
    {(-1, -1): 1, (0, -1): 1, (0, 1): 1, (1, 1): 1}
    >>> sign_determination([x**2 + 1, y], [x], x, y)
    {}
    """
    A = _algebra(system, gens)
    if A.empty:
        return {}
    positive, negative = _inertia(A.hermite(A.form()))
    total = positive - negative
    if total == 0:
        return {}
    signs: list[SignCondition] = [()]
    exponents: list[tuple[int, ...]] = [()]
    counts: Vector = [QQ(total)]
    forms: dict[tuple[int, ...], Vector] = {(): A.form()}
    for p in polys:
        transposed = A.transposed_multiplication(A.element(p))
        candidates = [s + (e,) for s in signs for e in (0, 1, -1)]
        rows = [a + (k,) for a in exponents for k in (0, 1, 2)]
        queries: Vector = []
        for a in rows:
            form = forms.get(a)
            if form is None:
                form = forms[a[:-1] + (a[-1] - 1,)] if a[-1] else forms[a[:-1]]
                if a[-1]:
                    form = A.compose(form, transposed)
                forms[a] = form
            positive, negative = _inertia(A.hermite(form))
            queries.append(QQ(positive - negative))
        matrix: Rows = [[QQ(_matrix_entry(a, s)) for s in candidates] for a in rows]
        solution = _solve(matrix, queries)
        kept = [k for k, c in enumerate(solution) if c]
        signs = [candidates[k] for k in kept]
        counts = [solution[k] for k in kept]
        restricted = [[row[k] for k in kept] for row in matrix]
        exponents = [rows[k] for k in _independent_rows(restricted)]
        # only the forms of the kept exponents are extended at the next step
        forms = {a: forms[a] for a in exponents}
    return dict(sorted((s, int(c)) for s, c in zip(signs, counts)))


def _matrix_entry(a: tuple[int, ...], s: SignCondition) -> int:
    value = 1
    for e, sign in zip(a, s):
        for _ in range(e):
            value *= sign
    return value


#: the relations a sign condition of ``lhs - rhs`` must satisfy, by class
_ALLOWED: dict[type[Relational], tuple[Sign, ...]] = {
    Eq: (0,), Ne: (1, -1), Gt: (1,), Ge: (0, 1), Lt: (-1,), Le: (0, -1)}


def decide_zero_dimensional(formula: Union[Boolean, bool], gens: Sequence[Symbol],
                            max_dimension: int = 64) -> Truth:
    r"""Whether a conjunction of polynomial relations with rational
    coefficients has a real solution, when its equations have finitely
    many complex solutions: by sign determination on the quotient algebra
    of the ideal of the equations, without any decomposition of the space.

    Parameters
    ==========

    formula : Boolean
        A relation or a conjunction of relations (``Eq``, ``Ne``, ``<``,
        ``<=``, ``>``, ``>=``) between polynomials in ``gens`` with rational
        coefficients.
    gens : Symbols
        The variables, taken to be real.
    max_dimension : int
        The largest dimension of the quotient algebra (the number of
        complex solutions with multiplicity) for which the decision is
        attempted.

    Returns
    =======

    ``True`` or ``False``, or ``None`` if the formula is not of this form
    (another connective, non-rational coefficients, equations with
    infinitely many solutions or too many).

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.hermite import decide_zero_dimensional
    >>> decide_zero_dimensional(Eq(x**2 + y**2, 1) & Eq(x*y, 1), [x, y])
    False
    >>> decide_zero_dimensional(Eq(x**2 + y**2, 4) & Eq(x*y, 1) & (x > y), [x, y])
    True
    >>> decide_zero_dimensional(Eq(x**2 + y**2, 1) & (x > y), [x, y]) is None
    True
    """
    if isinstance(formula, bool) or isinstance(formula, (BooleanTrue, BooleanFalse)):
        return bool(formula)
    atoms = formula.args if isinstance(formula, And) else (formula,)
    equations: list[Poly] = []
    sides: list[tuple[Poly, tuple[Sign, ...]]] = []
    for atom in atoms:
        allowed = _ALLOWED.get(type(atom)) if isinstance(atom, Relational) else None
        if allowed is None or not isinstance(atom, Relational):
            return None
        try:
            p = Poly(as_expr(atom.lhs) - as_expr(atom.rhs), *gens, domain=QQ)
        except (PolynomialError, CoercionFailed):
            return None
        if allowed == (0,):
            equations.append(p)
        else:
            sides.append((p, allowed))
    if len(equations) < len(gens):
        return None
    ideal = Ideal(equations, *gens)
    if ideal.is_whole_ring():
        return False
    if not ideal.is_zero_dimensional() or ideal.vector_space_dimension() > max_dimension:
        return None
    counts = sign_determination(ideal, [p for p, _ in sides])
    return any(all(s in allowed for s, (_, allowed) in zip(condition, sides)) for condition in counts)
