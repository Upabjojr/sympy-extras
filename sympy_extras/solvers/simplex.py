r"""Exact linear programming over the rationals, with certificates.

:func:`linear_program` maximises $c \cdot x$ subject to $A x \le b$ for
unrestricted rational $x$ by the two-phase simplex method in exact
arithmetic (Bland's rule, so that it terminates on degenerate problems),
and every answer comes with a certificate which is checked by a few
products, independently of the method:

* *optimal*: a point $x$ with $A x \le b$ and multipliers $y \ge 0$ with
  $y^T A = c$ and $y \cdot b = c \cdot x$ (strong duality: no feasible
  point does better, since $c \cdot x' = y^T A x' \le y \cdot b$);
* *infeasible*: multipliers $y \ge 0$ with $y^T A = 0$ and $y \cdot b <
  0$ (Farkas' lemma: the combination of the inequalities reads $0 \le y
  \cdot b < 0$);
* *unbounded*: a feasible point $x$ and a direction $d$ with $A d \le 0$
  and $c \cdot d > 0$.

:func:`linear_feasible` decides systems with strict inequalities too
($A_i x < b_i$ on chosen rows) by maximising a common slack, which is
the linear fragment of real arithmetic: no cylindrical decomposition is
needed for ``x > 1, x < 0``.

:func:`integer_program` maximises over the integer points by branch and
bound on the exact relaxations. It always terminates: by the proximity
theorem of Cook, Gerards, Schrijver and Tardos, if the integer program
has an optimum then one lies within $n \Delta$ (in every coordinate) of
any optimum of the relaxation, $\Delta$ the largest absolute value of a
subdeterminant of the integer matrix $A$, here bounded by Hadamard's
inequality; and a polyhedron with integer points has one with
coordinates at most $(n + 1) \Delta'$ for the matrix $[A\ b]$, which
decides the case of an unbounded relaxation. The search is confined to
these boxes.

:func:`eliminate_variables` projects a system of linear inequalities
(strict ones included) onto some of its unknowns by Fourier–Motzkin
elimination, which is quantifier elimination for the existential linear
formulas over the reals; the redundant inequalities the method produces
(doubly exponentially many otherwise) are removed after each variable by
linear programs.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.solvers.simplex import linear_program, linear_feasible
>>> result = linear_program([3, 2], [[1, 1], [1, 0], [0, 1], [-1, 0], [0, -1]], [4, 3, 3, 0, 0])
>>> result.status, result.value == 11, result.point == [3, 1]
('optimal', True, True)
>>> result.dual == [2, 1, 0, 0, 0]
True
>>> linear_program([1], [[1], [-1]], [0, -1]).status          # x <= 0, x >= 1
infeasible
>>> linear_program([1, 0], [[0, 1]], [1]).status
unbounded
>>> linear_feasible([[1], [-1]], [1, -1], [False, False]) == [1]         # x <= 1, x >= 1
True
>>> linear_feasible([[1], [-1]], [1, -1], [True, False]) is None        # x < 1, x >= 1
True

References
==========

.. [1] A. Schrijver, *Theory of Linear and Integer Programming*, Wiley,
       1986, Chapters 7 (Farkas' lemma, duality) and 11 (the simplex
       method).
.. [2] R. G. Bland, *New finite pivoting rules for the simplex method*,
       Mathematics of Operations Research 2 (1977), 103-107.
.. [3] V. Chvátal, *Linear Programming*, Freeman, 1983.
.. [4] W. Cook, A. M. H. Gerards, A. Schrijver, É. Tardos, *Sensitivity
       theorems in integer linear programming*, Mathematical Programming
       34 (1986), 251-264 (also [1], Theorem 17.2 and Corollary 17.1b).
.. [5] G. B. Dantzig, B. C. Eaves, *Fourier-Motzkin elimination and its
       dual*, Journal of Combinatorial Theory A 14 (1973), 288-297 (also
       [1], Section 12.2).
"""
from __future__ import annotations

from fractions import Fraction
from math import ceil, floor, gcd, isqrt
from typing import Final, Optional, Sequence, Union

__all__ = ['linear_program', 'linear_feasible', 'LinearProgramResult', 'integer_program', 'IntegerProgramResult',
           'LinearSystem', 'eliminate_variables']

#: a coefficient
Number = Union[int, Fraction]

_ZERO = Fraction(0)
_ONE = Fraction(1)


class LinearProgramResult:
    """The outcome of :func:`linear_program`.

    ``status`` is ``'optimal'``, ``'infeasible'`` or ``'unbounded'``.

    * optimal: ``value``, the maximiser ``point`` and the multipliers
      ``dual`` (``dual >= 0``, ``dual^T A = c``, ``dual . b = value``);
    * infeasible: ``dual`` is a Farkas certificate (``dual >= 0``,
      ``dual^T A = 0``, ``dual . b < 0``);
    * unbounded: a feasible ``point`` and a ``ray`` (``A ray <= 0``,
      ``c . ray > 0``).
    """

    def __init__(self, status: str, value: Optional[Fraction], point: Optional[list[Fraction]],
                 dual: Optional[list[Fraction]], ray: Optional[list[Fraction]]) -> None:
        self.status: Final[str] = status
        self.value: Final[Optional[Fraction]] = value
        self.point: Final[Optional[list[Fraction]]] = point
        self.dual: Final[Optional[list[Fraction]]] = dual
        self.ray: Final[Optional[list[Fraction]]] = ray

    def __repr__(self) -> str:
        return 'LinearProgramResult(%r, value=%s)' % (self.status, self.value)


class _Tableau:
    """A simplex tableau ``rows . z = rhs`` with a basis: the column
    ``basis[k]`` is the ``k``-th unit vector."""

    def __init__(self, rows: list[list[Fraction]], rhs: list[Fraction], basis: list[int]) -> None:
        self.rows = rows
        self.rhs = rhs
        self.basis = basis

    def pivot(self, k: int, j: int) -> None:
        row = self.rows[k]
        inverse = 1/row[j]
        self.rows[k] = row = [v*inverse for v in row]
        self.rhs[k] = self.rhs[k]*inverse
        for i in range(len(self.rows)):
            if i != k:
                factor = self.rows[i][j]
                if factor:
                    other = self.rows[i]
                    self.rows[i] = [a - factor*c for a, c in zip(other, row)]
                    self.rhs[i] -= factor*self.rhs[k]
        self.basis[k] = j

    def maximise(self, cost: list[Fraction], allowed: int) -> Optional[int]:
        """Simplex iterations with Bland's rule on the columns below
        ``allowed``; ``None`` at an optimum, the entering column of an
        unbounded direction otherwise."""
        while True:
            entering = -1
            for j in range(allowed):
                reduced = cost[j] - sum((cost[self.basis[k]]*self.rows[k][j] for k in range(len(self.rows))), _ZERO)
                if reduced > 0:
                    entering = j
                    break
            if entering < 0:
                return None
            leaving = -1
            best = _ZERO
            for k in range(len(self.rows)):
                coefficient = self.rows[k][entering]
                if coefficient > 0:
                    ratio = self.rhs[k]/coefficient
                    if leaving < 0 or ratio < best or (ratio == best and self.basis[k] < self.basis[leaving]):
                        leaving, best = k, ratio
            if leaving < 0:
                return entering
            self.pivot(leaving, entering)

    def multipliers(self, cost: list[Fraction], identity: list[int]) -> list[Fraction]:
        """``c_B B**-1``: the columns ``identity`` were the unit vectors
        of the initial tableau, and hold ``B**-1`` now."""
        return [sum((cost[self.basis[k]]*self.rows[k][column] for k in range(len(self.rows))), _ZERO)
                for column in identity]


def linear_program(c: Sequence[Number], A: Sequence[Sequence[Number]], b: Sequence[Number]) -> LinearProgramResult:
    """Maximise ``c . x`` subject to ``A x <= b``, the unknowns ``x``
    rational and of either sign.

    Parameters
    ==========

    c : the objective, one coefficient for each unknown.
    A, b : the rows of the constraints and their right-hand sides (an
        equation is two inequalities; ``x >= 0`` is the row ``-x <= 0``).

    Returns
    =======

    A :class:`LinearProgramResult` with the status and its certificate.

    Examples
    ========

    >>> from sympy_extras.solvers.simplex import linear_program
    >>> r = linear_program([1, 1], [[2, 1], [1, 3], [-1, 0], [0, -1]], [4, 6, 0, 0])
    >>> r.status, r.value, r.point
    ('optimal', Fraction(14, 5), [Fraction(6, 5), Fraction(8, 5)])
    >>> r.dual
    [Fraction(2, 5), Fraction(1, 5), Fraction(0, 1), Fraction(0, 1)]
    >>> farkas = linear_program([0, 0], [[1, 1], [-1, 0], [0, -1]], [-1, 0, 0])
    >>> farkas.status, farkas.dual
    ('infeasible', [Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)])
    """
    n = len(c)
    m = len(A)
    if len(b) != m:
        raise ValueError('one right-hand side for each row')
    for row in A:
        if len(row) != n:
            raise ValueError('the rows must have %d coefficients' % n)
    objective = [Fraction(v) for v in c]
    matrix = [[Fraction(v) for v in row] for row in A]
    rhs = [Fraction(v) for v in b]
    # columns: x+ (n), x- (n), slacks (m), artificials (one for each row
    # whose right-hand side is negative)
    signs = [1 if value >= 0 else -1 for value in rhs]
    artificial_rows = [i for i in range(m) if signs[i] < 0]
    width = 2*n + m + len(artificial_rows)
    rows: list[list[Fraction]] = []
    basis: list[int] = []
    identity: list[int] = []
    for i in range(m):
        line = [_ZERO]*width
        for j in range(n):
            line[j] = signs[i]*matrix[i][j]
            line[n + j] = -signs[i]*matrix[i][j]
        line[2*n + i] = Fraction(signs[i])
        if signs[i] < 0:
            column = 2*n + m + artificial_rows.index(i)
            line[column] = _ONE
        else:
            column = 2*n + i
        rows.append(line)
        basis.append(column)
        identity.append(column)
    tableau = _Tableau(rows, [signs[i]*rhs[i] for i in range(m)], basis)
    real_columns = 2*n + m

    def point() -> list[Fraction]:
        z = [_ZERO]*width
        for k, column in enumerate(tableau.basis):
            z[column] = tableau.rhs[k]
        return [z[j] - z[n + j] for j in range(n)]

    def original(multipliers: list[Fraction]) -> list[Fraction]:
        return [multipliers[i]*signs[i] for i in range(m)]

    if artificial_rows:
        first = [_ZERO]*real_columns + [-_ONE]*len(artificial_rows)
        tableau.maximise(first, width)
        infeasibility = sum((tableau.rhs[k] for k in range(m) if tableau.basis[k] >= real_columns), _ZERO)
        if infeasibility > 0:
            return LinearProgramResult('infeasible', None, None, original(tableau.multipliers(first, identity)), None)
        # artificial variables left in the basis at level zero go out
        for k in range(m):
            if tableau.basis[k] >= real_columns:
                for j in range(real_columns):
                    if tableau.rows[k][j] != 0:
                        tableau.pivot(k, j)
                        break
    cost = objective + [-v for v in objective] + [_ZERO]*(m + len(artificial_rows))
    entering = tableau.maximise(cost, real_columns)
    if entering is not None:
        direction = [_ZERO]*width
        direction[entering] = _ONE
        for k, column in enumerate(tableau.basis):
            direction[column] = -tableau.rows[k][entering]
        return LinearProgramResult('unbounded', None, point(), None,
                                   [direction[j] - direction[n + j] for j in range(n)])
    x = point()
    value = sum((objective[j]*x[j] for j in range(n)), _ZERO)
    return LinearProgramResult('optimal', value, x, original(tableau.multipliers(cost, identity)), None)


def linear_feasible(A: Sequence[Sequence[Number]], b: Sequence[Number], strict: Optional[Sequence[bool]] = None
                    ) -> Optional[list[Fraction]]:
    """A rational point with ``A_i x <= b_i`` on every row, strictly
    (``<``) on the rows marked in ``strict``; ``None`` when there is
    none.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.solvers.simplex import linear_feasible
    >>> linear_feasible([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], [True, True, True])      # the open triangle
    [Fraction(1, 3), Fraction(1, 3)]
    >>> linear_feasible([[1, 1], [-1, -1]], [1, -1], [True, False]) is None      # x + y < 1, x + y >= 1
    True
    """
    m = len(A)
    flags = [False]*m if strict is None else [bool(v) for v in strict]
    if len(flags) != m or len(b) != m:
        raise ValueError('one right-hand side and one flag for each row')
    n = len(A[0]) if m else 0
    if not any(flags):
        result = linear_program([0]*n, A, b)
        return result.point if result.status == 'optimal' else None
    # maximise t with A_i x + t <= b_i on the strict rows and t <= 1
    rows: list[list[Fraction]] = [[Fraction(v) for v in row] + [_ONE if flag else _ZERO] for row, flag in zip(A, flags)]
    rows.append([_ZERO]*n + [_ONE])
    bounds: list[Fraction] = [Fraction(v) for v in b] + [_ONE]
    result = linear_program([_ZERO]*n + [_ONE], rows, bounds)
    if result.status != 'optimal' or result.point is None or result.value is None or result.value <= 0:
        return None
    return result.point[:n]


# ---------------------------------------------------------------------------
# integer programming

class IntegerProgramResult:
    """The outcome of :func:`integer_program`: ``status`` is
    ``'optimal'`` (with ``value`` and the integer ``point``),
    ``'infeasible'`` (no integer point) or ``'unbounded'`` (with an
    integer ``point`` of the polyhedron)."""

    def __init__(self, status: str, value: Optional[Fraction], point: Optional[list[int]]) -> None:
        self.status: Final[str] = status
        self.value: Final[Optional[Fraction]] = value
        self.point: Final[Optional[list[int]]] = point

    def __repr__(self) -> str:
        return 'IntegerProgramResult(%r, value=%s, point=%s)' % (self.status, self.value, self.point)


def _integer_rows(rows: list[list[Fraction]]) -> list[list[int]]:
    """Each row multiplied by the common denominator of its entries."""
    result: list[list[int]] = []
    for row in rows:
        common = 1
        for v in row:
            common = common*v.denominator//gcd(common, v.denominator)
        result.append([int(v*common) for v in row])
    return result


def _hadamard(rows: list[list[int]]) -> int:
    """A bound on the absolute values of the subdeterminants of an
    integer matrix: the product of the Euclidean lengths of its non-zero
    rows (each at least one), rounded up."""
    bound = 1
    for row in rows:
        norm = sum(v*v for v in row)
        if norm:
            bound *= isqrt(norm - 1) + 1
    return bound


def _branch_and_bound(c: list[Fraction], A: list[list[Fraction]], b: list[Fraction], low: list[int], high: list[int]
                      ) -> Optional[tuple[Fraction, list[int]]]:
    """The best integer point of ``A x <= b`` in the box ``low <= x <=
    high`` for the objective ``c``, with its value; ``None`` when the
    box has no integer point of the polyhedron."""
    n = len(c)
    best: Optional[tuple[Fraction, list[int]]] = None
    pending: list[tuple[list[int], list[int]]] = [(low, high)]
    while pending:
        lower, upper = pending.pop()
        if any(l > u for l, u in zip(lower, upper)):
            continue
        rows = [list(row) for row in A]
        bounds = list(b)
        for j in range(n):
            rows.append([_ONE if k == j else _ZERO for k in range(n)])
            bounds.append(Fraction(upper[j]))
            rows.append([-_ONE if k == j else _ZERO for k in range(n)])
            bounds.append(Fraction(-lower[j]))
        relaxed = linear_program(c, rows, bounds)
        if relaxed.status != 'optimal' or relaxed.point is None or relaxed.value is None:
            continue
        if best is not None and relaxed.value <= best[0]:
            continue
        fractional = next((j for j in range(n) if relaxed.point[j].denominator != 1), -1)
        if fractional < 0:
            best = (relaxed.value, [int(v) for v in relaxed.point])
            continue
        value = relaxed.point[fractional]
        down = list(upper)
        down[fractional] = floor(value)
        up = list(lower)
        up[fractional] = ceil(value)
        pending.append((lower, down))
        pending.append((up, upper))
    return best


def integer_program(c: Sequence[Number], A: Sequence[Sequence[Number]], b: Sequence[Number]) -> IntegerProgramResult:
    """Maximise ``c . x`` subject to ``A x <= b`` over the integer
    vectors ``x`` (of either sign), by branch and bound on exact linear
    relaxations inside a box which is proved to contain an optimum.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.solvers.simplex import integer_program
    >>> integer_program([5, 4], [[6, 4], [1, 2], [-1, 0], [0, -1]], [24, 6, 0, 0])
    IntegerProgramResult('optimal', value=20, point=[4, 0])
    >>> integer_program([1, 1], [[2, 2], [-2, -2]], [1, -1]).status           # x + y = 1/2
    infeasible
    >>> integer_program([0, 0], [[3, -3], [-3, 3]], [2, -1]).status           # 1/3 <= x - y <= 2/3
    infeasible
    >>> integer_program([1, 0], [[0, 1], [0, -1]], [Fraction(5, 2), 0]).status
    unbounded
    """
    n = len(c)
    objective = [Fraction(v) for v in c]
    matrix = [[Fraction(v) for v in row] for row in A]
    rhs = [Fraction(v) for v in b]
    relaxed = linear_program(objective, matrix, rhs)
    if relaxed.status == 'infeasible':
        return IntegerProgramResult('infeasible', None, None)
    if relaxed.status == 'unbounded':
        # an integer point, if there is one, within (n + 1) Delta([A b])
        radius = (n + 1)*_hadamard(_integer_rows([row + [value] for row, value in zip(matrix, rhs)]))
        found = _branch_and_bound([_ZERO]*n, matrix, rhs, [-radius]*n, [radius]*n)
        if found is None:
            return IntegerProgramResult('infeasible', None, None)
        return IntegerProgramResult('unbounded', None, found[1])
    if relaxed.point is None:
        raise ArithmeticError('an optimum without a point')
    # an integer optimum, if there is one, within n Delta(A) of the
    # optimum of the relaxation
    radius = n*_hadamard(_integer_rows(matrix))
    low = [floor(v) - radius for v in relaxed.point]
    high = [ceil(v) + radius for v in relaxed.point]
    found = _branch_and_bound(objective, matrix, rhs, low, high)
    if found is None:
        return IntegerProgramResult('infeasible', None, None)
    return IntegerProgramResult('optimal', found[0], found[1])


# ---------------------------------------------------------------------------
# Fourier-Motzkin elimination

class LinearSystem:
    """The inequalities ``A_i x <= b_i``, strict (``<``) on the rows
    marked in ``strict``."""

    def __init__(self, A: list[list[Fraction]], b: list[Fraction], strict: list[bool]) -> None:
        self.A: Final[list[list[Fraction]]] = A
        self.b: Final[list[Fraction]] = b
        self.strict: Final[list[bool]] = strict

    def contains(self, point: Sequence[Number]) -> bool:
        """Whether the point satisfies every inequality."""
        for row, bound, flag in zip(self.A, self.b, self.strict):
            value = sum((a*Fraction(v) for a, v in zip(row, point)), _ZERO)
            if value > bound or (flag and value == bound):
                return False
        return True

    def is_feasible(self) -> bool:
        """Whether the system has a solution."""
        if not self.A:
            return True
        return linear_feasible(self.A, self.b, self.strict) is not None

    def __repr__(self) -> str:
        return 'LinearSystem(%r, %r, %r)' % (self.A, self.b, self.strict)


#: an inequality: the coefficients, the bound, whether it is strict
_Row = tuple[tuple[Fraction, ...], Fraction, bool]


def _normal_row(coefficients: Sequence[Fraction], bound: Fraction, strict: bool) -> _Row:
    """The row divided by the absolute value of its first non-zero
    coefficient (of the bound for a row without unknowns)."""
    scale = next((abs(v) for v in coefficients if v), _ZERO)
    if scale == 0:
        scale = abs(bound) if bound else _ONE
    return tuple(v/scale for v in coefficients), bound/scale, strict


def _without_redundant(rows: list[_Row]) -> list[_Row]:
    """The rows without duplicates, without those implied by a row with
    the same left-hand side, without the trivial ones, and without
    those which a linear program proves implied by the others."""
    strongest: dict[tuple[Fraction, ...], tuple[Fraction, bool]] = {}
    for coefficients, bound, strict in rows:
        if not any(coefficients):
            if bound > 0 or (bound == 0 and not strict):
                continue
            # 0 <= negative, or 0 < 0: the system is contradictory
            return [(coefficients, -_ONE, False)]
        current = strongest.get(coefficients)
        if current is None or bound < current[0] or (bound == current[0] and strict and not current[1]):
            strongest[coefficients] = (bound, strict)
    kept: list[_Row] = [(coefficients, bound, strict) for coefficients, (bound, strict) in strongest.items()]
    index = 0
    while index < len(kept) and len(kept) > 1:
        coefficients, bound, strict = kept[index]
        others = kept[:index] + kept[index + 1:]
        result = linear_program(list(coefficients), [list(row[0]) for row in others], [row[1] for row in others])
        if result.status == 'infeasible':
            return [(tuple(_ZERO for _ in coefficients), -_ONE, False)]
        # the supremum over the closure of the others: below the bound,
        # the row says nothing new (at the bound, a strict row still does)
        if result.status == 'optimal' and result.value is not None and (
                result.value < bound or (result.value == bound and not strict)):
            kept = others
        else:
            index += 1
    return kept


def eliminate_variables(A: Sequence[Sequence[Number]], b: Sequence[Number], variables: Sequence[int],
                        strict: Optional[Sequence[bool]] = None) -> LinearSystem:
    """The projection of ``{x: A x <= b}`` (strict on the rows marked in
    ``strict``) along the unknowns ``variables``: a system in the other
    unknowns, in their order, satisfied by exactly the points which
    extend to a solution.

    For each eliminated unknown, the rows where it has a positive
    coefficient (upper bounds) are combined with those where it is
    negative (lower bounds), a combination being strict when one of the
    two is; the rows a linear program proves redundant are dropped.

    Examples
    ========

    >>> from sympy_extras.solvers.simplex import eliminate_variables
    >>> triangle = [[-1, 0], [0, -1], [1, 1]]             # x >= 0, y >= 0, x + y <= 1
    >>> shadow = eliminate_variables(triangle, [0, 0, 1], [1])
    >>> sorted(zip(shadow.A, shadow.b)) == [([-1], 0), ([1], 1)]             # 0 <= x <= 1
    True
    >>> open_one = eliminate_variables(triangle, [0, 0, 1], [1], [False, True, False])     # y > 0
    >>> open_one.contains([1]), open_one.contains([Fraction(1, 2)])
    (False, True)
    >>> eliminate_variables([[1], [-1]], [0, -1], [0]).is_feasible()         # exists x: x <= 0, x >= 1
    False
    """
    m = len(A)
    n = len(A[0]) if m else 0
    flags = [False]*m if strict is None else [bool(v) for v in strict]
    if len(b) != m or len(flags) != m:
        raise ValueError('one right-hand side and one flag for each row')
    for j in variables:
        if not 0 <= j < n:
            raise ValueError('no unknown number %d' % j)
    rows: list[_Row] = [_normal_row([Fraction(v) for v in row], Fraction(bound), flag)
                        for row, bound, flag in zip(A, b, flags)]
    for j in sorted(set(variables)):
        upper = [row for row in rows if row[0][j] > 0]
        lower = [row for row in rows if row[0][j] < 0]
        combined: list[_Row] = [row for row in rows if row[0][j] == 0]
        for p in upper:
            for q in lower:
                wp, wq = 1/p[0][j], -1/q[0][j]
                coefficients = [wp*u + wq*v for u, v in zip(p[0], q[0])]
                coefficients[j] = _ZERO
                combined.append(_normal_row(coefficients, wp*p[1] + wq*q[1], p[2] or q[2]))
        rows = _without_redundant(combined)
    remaining = [j for j in range(n) if j not in set(variables)]
    return LinearSystem([[row[0][j] for j in remaining] for row in rows], [row[1] for row in rows],
                        [row[2] for row in rows])
