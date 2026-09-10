"""Linear problems over the integers: Diophantine systems, nonnegative
solutions and Presburger arithmetic.

* :func:`linear_diophantine_system` gives the general integer solution of
  a system of linear equations with integer coefficients through a
  Hermite normal form with its unimodular transformation (the method
  Wolfram's ``Reduce`` uses for linear Diophantine equations); SymPy's
  ``diophantine`` handles one linear equation at a time.
* :func:`minimal_nonnegative_solutions` and :func:`hilbert_basis` give the
  minimal solutions in nonnegative integers of a linear system (the
  Hilbert basis of the solution monoid for a homogeneous one) by the
  algorithm of Contejean and Devie, a breadth-first completion in which a
  candidate is extended in a coordinate only when that decreases its
  defect.
* :func:`cooper` eliminates an existential quantifier from a formula of
  Presburger arithmetic (linear relations and divisibility conditions on
  integer variables) by Cooper's algorithm, which is used by
  :func:`sympy_extras.assumptions.resolve` over the integers.

References
==========

.. [ContejeanDevie] E. Contejean, H. Devie, An efficient incremental
   algorithm for solving systems of linear Diophantine equations,
   Information and Computation 113 (1994).
.. [Cooper] D. C. Cooper, Theorem proving in arithmetic without
   multiplication, Machine Intelligence 7 (1972).
.. [Cohen] H. Cohen, A Course in Computational Algebraic Number Theory,
   Springer 1993, section 2.4 (Hermite normal form).
"""
from __future__ import annotations

from math import gcd
from typing import Optional, Sequence, Union

from sympy.core.expr import Expr
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.relational import Relational, Eq, Ne, Lt, Le, Gt, Ge
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import (Boolean, BooleanTrue, BooleanFalse, And, Or, Not,
    Implies, Equivalent, Xor, ITE, true, false)
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly

from sympy_extras._typing import QuantifierPrefix, as_boolean, as_expr, free_symbols

__all__ = ['hermite_normal_form_with_transform', 'linear_diophantine_system',
    'hilbert_basis', 'minimal_nonnegative_solutions', 'cooper',
    'presburger_quantifier_elimination', 'is_presburger']

#: an integer matrix as a list of rows
IntMatrix = list[list[int]]
#: an integer vector
IntVector = list[int]


# ---------------------------------------------------------------------------
# Hermite normal form and linear systems

def hermite_normal_form_with_transform(A: IntMatrix) -> tuple[IntMatrix, IntMatrix]:
    """The column Hermite normal form ``H`` of an integer matrix and the
    unimodular ``U`` with ``A U = H``: ``H`` is lower triangular in
    echelon form, with positive pivots and the entries to the left of a
    pivot reduced modulo it.

    Examples
    ========

    >>> from sympy_extras.solvers.integers import hermite_normal_form_with_transform
    >>> H, U = hermite_normal_form_with_transform([[2, 4, 6], [3, 1, 5]])
    >>> H
    [[2, 0, 0], [0, 1, 0]]
    """
    m = len(A)
    n = len(A[0]) if A else 0
    H = [list(row) for row in A]
    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def combine(j: int, k: int, p: int, q: int, r: int, s: int) -> None:
        # columns j, k <- p*col_j + q*col_k, r*col_j + s*col_k
        for rows in (H, U):
            for row in rows:
                cj, ck = row[j], row[k]
                row[j], row[k] = p*cj + q*ck, r*cj + s*ck

    pivot_column = 0
    for i in range(m):
        if pivot_column >= n:
            break
        # clear the entries of row i to the right of the pivot column
        for k in range(pivot_column + 1, n):
            if H[i][k] == 0:
                continue
            a, b = H[i][pivot_column], H[i][k]
            g, u, v = _extended_gcd(a, b)
            combine(pivot_column, k, u, v, -b//g, a//g)
        if H[i][pivot_column] == 0:
            continue
        if H[i][pivot_column] < 0:
            for rows in (H, U):
                for row in rows:
                    row[pivot_column] = -row[pivot_column]
        # reduce the entries to the left of the pivot
        p = H[i][pivot_column]
        for k in range(pivot_column):
            q = H[i][k]//p
            if q:
                for rows in (H, U):
                    for row in rows:
                        row[k] -= q*row[pivot_column]
        pivot_column += 1
    return H, U


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """``(g, u, v)`` with ``g = gcd(a, b) = u a + v b``, ``g >= 0``."""
    old_r, r = a, b
    old_u, u = 1, 0
    old_v, v = 0, 1
    while r:
        q = old_r//r
        old_r, r = r, old_r - q*r
        old_u, u = u, old_u - q*u
        old_v, v = v, old_v - q*v
    if old_r < 0:
        return -old_r, -old_u, -old_v
    return old_r, old_u, old_v


def _linear_system(equations: Sequence[Expr], symbols: Sequence[Symbol]) -> Optional[tuple[IntMatrix, IntVector]]:
    """``A``, ``b`` with the equations ``A x = b`` (integer coefficients)."""
    A: IntMatrix = []
    b: IntVector = []
    if not symbols:
        raise ValueError("no unknown to solve for")
    for e in equations:
        if not free_symbols(e) <= set(symbols):
            return None
        try:
            poly = Poly(e, *symbols)
        except PolynomialError:
            return None
        if poly.total_degree() > 1:
            return None
        row: IntVector = []
        for s in symbols:
            c = poly.coeff_monomial(s)
            if not isinstance(c, Integer):
                return None
            row.append(int(c))
        constant = poly.coeff_monomial(1)
        if not isinstance(constant, Integer):
            return None
        A.append(row)
        b.append(-int(constant))
    return A, b


def linear_diophantine_system(equations: Sequence[Union[Expr, Eq]], symbols: Sequence[Symbol]
                              ) -> Optional[tuple[list[Expr], list[Symbol]]]:
    """The general integer solution of a system of linear equations with
    integer coefficients: the values of the symbols as affine expressions
    in free integer parameters, and the parameters; ``None`` when the
    system has no integer solution.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.solvers.integers import linear_diophantine_system
    >>> values, parameters = linear_diophantine_system([Eq(3*x + 5*y, 7)], [x, y])
    >>> values
    [14 - 5*t0, 3*t0 - 7]
    >>> linear_diophantine_system([Eq(2*x + 4*y, 3)], [x, y]) is None
    True
    >>> values, parameters = linear_diophantine_system([x + y + z - 6, x - y - 2], [x, y, z])
    >>> values
    [t0, t0 - 2, 8 - 2*t0]
    """
    exprs = [as_expr(e.lhs - e.rhs) if isinstance(e, Eq) else as_expr(e) for e in equations]
    system = _linear_system(exprs, symbols)
    if system is None:
        raise ValueError("a linear system with integer coefficients is expected")
    A, b = system
    n = len(symbols)
    H, U = hermite_normal_form_with_transform(A)
    # solve H y = b by forward substitution; the columns of H which are
    # zero leave their y free
    y: list[Optional[int]] = [None]*n
    rank = 0
    for i in range(len(A)):
        pivot = next((j for j in range(n) if H[i][j] != 0 and all(H[r][j] == 0 for r in range(i))), None)
        residual = b[i]
        for j in range(n):
            value = y[j]
            if value is not None:
                residual -= H[i][j]*value
        if pivot is None:
            if residual != 0:
                return None
            continue
        if residual % H[i][pivot] != 0:
            return None
        y[pivot] = residual//H[i][pivot]
        rank += 1
    parameters = [Symbol('t%d' % k, integer=True) for k in range(n - rank)]
    free_index = 0
    y_values: list[Expr] = []
    for j in range(n):
        if y[j] is None:
            y_values.append(as_expr(parameters[free_index]))
            free_index += 1
        else:
            y_values.append(as_expr(Integer(y[j])))
    values = [as_expr(sum((Integer(U[i][j])*y_values[j] for j in range(n)), S.Zero)) for i in range(n)]
    return values, parameters


# ---------------------------------------------------------------------------
# Contejean-Devie

def _defect(A: IntMatrix, v: IntVector) -> IntVector:
    return [sum(row[j]*v[j] for j in range(len(v))) for row in A]


def _dominates(v: IntVector, w: IntVector) -> bool:
    """``v >= w`` componentwise."""
    return all(a >= b for a, b in zip(v, w))


def hilbert_basis(A: IntMatrix, limit: int = 10000) -> list[IntVector]:
    """The minimal nonzero solutions in nonnegative integers of the
    homogeneous system ``A x = 0`` (Contejean–Devie), which generate all
    its nonnegative solutions.

    Examples
    ========

    >>> from sympy_extras.solvers.integers import hilbert_basis
    >>> hilbert_basis([[1, 1, -1]])
    [[0, 1, 1], [1, 0, 1]]
    >>> hilbert_basis([[2, -3]])
    [[3, 2]]
    """
    n = len(A[0]) if A else 0
    solutions: list[IntVector] = []
    frontier: list[IntVector] = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    seen: set[tuple[int, ...]] = set(tuple(v) for v in frontier)
    examined = 0
    while frontier:
        new_frontier: list[IntVector] = []
        for v in frontier:
            examined += 1
            if examined > limit:
                raise RuntimeError("the completion did not finish within the limit")
            d = _defect(A, v)
            if all(c == 0 for c in d):
                if not any(_dominates(v, s) for s in solutions):
                    solutions.append(v)
                continue
            if any(_dominates(v, s) for s in solutions):
                continue
            for i in range(n):
                # extend in the directions which decrease the defect
                if sum(d[r]*A[r][i] for r in range(len(A))) < 0:
                    w = list(v)
                    w[i] += 1
                    key = tuple(w)
                    if key not in seen:
                        seen.add(key)
                        new_frontier.append(w)
        frontier = new_frontier
    return sorted(solutions)


def minimal_nonnegative_solutions(A: IntMatrix, b: IntVector, limit: int = 10000
                                  ) -> tuple[list[IntVector], list[IntVector]]:
    """The minimal solutions in nonnegative integers of ``A x = b`` and
    the Hilbert basis of ``A x = 0``: every nonnegative solution is one of
    the former plus a combination of the latter.

    Examples
    ========

    >>> from sympy_extras.solvers.integers import minimal_nonnegative_solutions
    >>> minimal_nonnegative_solutions([[3, 5]], [22])
    ([[4, 2]], [])
    >>> minimal_nonnegative_solutions([[3, -5]], [1])
    ([[2, 1]], [[5, 3]])
    >>> minimal_nonnegative_solutions([[1, 1]], [2])
    ([[0, 2], [1, 1], [2, 0]], [])
    """
    n = len(A[0]) if A else 0
    homogenized = [row + [-c] for row, c in zip(A, b)]
    basis = hilbert_basis(homogenized, limit)
    particular = sorted(v[:n] for v in basis if v[n] == 1)
    homogeneous = sorted(v[:n] for v in basis if v[n] == 0)
    return particular, homogeneous


# ---------------------------------------------------------------------------
# Cooper's algorithm

class _Linear:
    """``coefficient * x + rest`` with ``rest`` free of ``x``."""

    def __init__(self, coefficient: int, rest: Expr) -> None:
        self.coefficient = coefficient
        self.rest = rest


def _split_linear(e: Expr, x: Symbol) -> Optional[_Linear]:
    if not e.has(x):
        return _Linear(0, e)
    try:
        poly = Poly(e, x)
    except PolynomialError:
        return None
    if poly.degree() != 1:
        return None
    a, b = poly.all_coeffs()
    if not isinstance(a, Integer):
        return None
    return _Linear(int(a), as_expr(b))


def _divisibility(atom: Boolean) -> Optional[tuple[int, Expr]]:
    """``(k, e)`` for an atom ``Eq(Mod(e, k), r)`` or ``Ne(Mod(e, k), r)``
    with an integer residue ``r``, read as ``k | e - r`` (``Mod`` may
    carry an integer factor, extracted by SymPy from the argument:
    ``c Mod(e, k) = 0`` is ``k | e`` again). A residue outside
    ``0 <= r < k`` is left alone: the atom is then simply false or true,
    which :func:`_canonical_divisibilities` settles (sympy-extras#44)."""
    if isinstance(atom, (Eq, Ne)) and isinstance(atom.rhs, Integer):
        lhs, residue = atom.lhs, int(atom.rhs)
        if residue == 0 and isinstance(lhs, Mul) and len(lhs.args) == 2 \
                and isinstance(lhs.args[0], Integer) and lhs.args[0] != 0:
            lhs = lhs.args[1]
        if isinstance(lhs, Mod):
            e, k = lhs.args
            if isinstance(k, Integer) and int(k) > 0 and 0 <= residue < int(k):
                return int(k), as_expr(e - residue)
    return None


def _impossible_residue(atom: Boolean) -> Optional[Boolean]:
    """``Mod(e, k) = r`` with ``r`` outside ``[0, k)`` never holds, and its
    negation always does."""
    if isinstance(atom, (Eq, Ne)) and isinstance(atom.rhs, Integer) and isinstance(atom.lhs, Mod):
        k = atom.lhs.args[1]
        if isinstance(k, Integer) and int(k) > 0 and not 0 <= int(atom.rhs) < int(k):
            return false if isinstance(atom, Eq) else true
    return None


def _divides(kind: type, k: int, e: Expr) -> Boolean:
    """The atom ``k | e`` (``kind`` is ``Eq``) or its negation (``Ne``)."""
    if k == 1:
        return true if kind is Eq else false
    return as_boolean(kind(Mod(e, k), 0))


def _canonical_divisibilities(formula: Boolean) -> Boolean:
    """The divisibility atoms rewritten by :func:`_divides`."""
    replacements: dict[Boolean, Boolean] = {}
    for atom in formula.atoms(Relational):
        divisibility = _divisibility(atom)
        if divisibility is not None:
            replacements[atom] = _divides(type(atom), *divisibility)
            continue
        impossible = _impossible_residue(atom)
        if impossible is not None:
            replacements[atom] = impossible
    return _substitute(formula, replacements)


def is_presburger(formula: Boolean, integers: set[Symbol]) -> bool:
    """Whether the formula is a Boolean combination of linear relations
    with integer coefficients and divisibility conditions
    ``Eq(Mod(e, k), 0)`` in the given integer variables."""
    for atom in formula.atoms(Relational):
        divisibility = _divisibility(atom)
        if divisibility is not None:
            if not _integer_linear(divisibility[1], integers):
                return False
            continue
        if _impossible_residue(atom) is not None:
            continue
        if not _integer_linear(as_expr(atom.lhs - atom.rhs), integers):
            return False
    return True


def _integer_linear(e: Expr, integers: set[Symbol]) -> bool:
    symbols = free_symbols(e)
    if not symbols <= integers:
        return False
    if not symbols:
        return isinstance(e, Integer)
    try:
        poly = Poly(e, *sorted(symbols, key=lambda s: s.name))
    except PolynomialError:
        return False
    return poly.total_degree() <= 1 and all(isinstance(c, Integer) for c in poly.coeffs())


def _lcm(values: list[int]) -> int:
    result = 1
    for v in values:
        result = result*v//gcd(result, v)
    return result


def _substitute(formula: Boolean, replacements: dict[Boolean, Boolean]) -> Boolean:
    if formula in replacements:
        return replacements[formula]
    if isinstance(formula, (And, Or, Not, Implies, Equivalent, Xor, ITE)):
        return as_boolean(formula.func(*[_substitute(as_boolean(a), replacements) for a in formula.args]))
    return formula


def _normalize_coefficients(formula: Boolean, x: Symbol) -> tuple[Boolean, int]:
    """Rewrite the atoms so that ``x`` has coefficient ``+1`` or ``-1``:
    each atom is multiplied by ``m/|a|`` with ``m`` the least common
    multiple of the coefficients, and the divisibility ``m | x`` is added
    (for the new variable ``x' = m x``)."""
    atoms = [a for a in formula.atoms(Relational)]
    coefficients: list[int] = []
    parts: dict[Relational, _Linear] = {}
    for atom in atoms:
        divisibility = _divisibility(atom)
        e = divisibility[1] if divisibility is not None else as_expr(atom.lhs - atom.rhs)
        lin = _split_linear(e, x)
        if lin is None:
            raise ValueError("%s is not linear in %s" % (atom, x))
        parts[atom] = lin
        if lin.coefficient:
            coefficients.append(abs(lin.coefficient))
    m = _lcm(coefficients) if coefficients else 1
    replacements: dict[Boolean, Boolean] = {}
    for atom, lin in parts.items():
        if lin.coefficient == 0:
            continue
        factor = m//abs(lin.coefficient)
        sign = 1 if lin.coefficient > 0 else -1
        # a x + b rel 0  <=>  sign*(m x' ... ) : with x' = m x, a x + b = sign*(x' + factor*sign*b)/...
        # multiply by factor > 0: factor*a*x + factor*b = sign*x' + factor*b
        new_rest = as_expr(factor*lin.rest)
        divisibility = _divisibility(atom)
        if divisibility is not None:
            k = divisibility[0]*factor
            replacements[atom] = _divides(type(atom), k, as_expr(sign*x + new_rest))
        else:
            replacements[atom] = as_boolean(type(atom)(sign*x + new_rest, 0))
    return _substitute(formula, replacements), m


def cooper(formula: Boolean, x: Symbol) -> Boolean:
    """`\\exists x \\in \\mathbb{Z}\\, \\varphi` for a formula of Presburger
    arithmetic, as a quantifier-free formula in the other integer
    variables (with divisibility conditions ``Eq(Mod(e, k), 0)``).

    Examples
    ========

    >>> from sympy import Eq, Mod
    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.integers import cooper
    >>> cooper(Eq(2*x, y), x)
    Eq(Mod(y, 2), 0)
    >>> cooper((x > y) & (x < y + 2), x)
    True
    >>> cooper((x > y) & (x < y + 1), x)
    False
    >>> cooper((3*x > y) & (3*x < y + 3), x)
    Eq(Mod(y + 1, 3), 0) | Eq(Mod(y + 2, 3), 0)
    """
    # negations are pushed down to the atoms: the argument below needs a
    # formula which is monotone in its atoms
    formula = as_boolean(as_boolean(formula).to_nnf(simplify=False))
    # residues first: Mod(e, k) = r is k | e - r, and one outside [0, k)
    # is a constant (sympy-extras#44); the atom loop below reads the
    # canonical form only
    formula = _canonical_divisibilities(formula)
    if not formula.has(x):
        return formula
    normalized, m = _normalize_coefficients(formula, x)
    if m != 1:
        normalized = And(normalized, Eq(Mod(x, m), 0))
    atoms = [a for a in normalized.atoms(Relational)]
    parts: dict[Relational, _Linear] = {}
    moduli: list[int] = []
    # the lower thresholds: for each atom, the greatest value of ``x``
    # below which the atom is false and above which it is true (the
    # relations ``x = c`` and ``x != c`` seen as ``c - 1 < x < c + 1`` and
    # ``x < c or c < x``)
    thresholds: list[Expr] = []
    for atom in atoms:
        divisibility = _divisibility(atom)
        e = divisibility[1] if divisibility is not None else as_expr(atom.lhs - atom.rhs)
        lin = _split_linear(e, x)
        if lin is None:
            raise ValueError("%s is not linear in %s" % (atom, x))
        parts[atom] = lin
        if divisibility is not None:
            if lin.coefficient:
                moduli.append(divisibility[0])
            continue
        if lin.coefficient == 0:
            continue
        kind = type(atom)
        if lin.coefficient == 1:
            # x + r rel 0
            c = as_expr(-lin.rest)
            if kind is Gt:
                thresholds.append(c)
            elif kind is Ge:
                thresholds.append(as_expr(c - 1))
        else:
            # -x + r rel 0
            c = lin.rest
            if kind is Lt:
                thresholds.append(c)
            elif kind is Le:
                thresholds.append(as_expr(c - 1))
        if kind is Eq:
            thresholds.append(as_expr(c - 1))
        elif kind is Ne:
            thresholds.append(c)
    delta = _lcm(moduli) if moduli else 1
    disjuncts: list[Boolean] = []
    # the formula at -oo: the bounds settle, the divisibilities are periodic
    for j in range(1, delta + 1):
        replacements: dict[Boolean, Boolean] = {}
        for atom, lin in parts.items():
            if lin.coefficient == 0:
                continue
            if _divisibility(atom) is not None:
                replacements[atom] = as_boolean(atom.subs(x, j))
            else:
                kind = type(atom)
                if kind is Eq:
                    replacements[atom] = false
                elif kind is Ne:
                    replacements[atom] = true
                elif kind in (Lt, Le):
                    replacements[atom] = true if lin.coefficient == 1 else false
                else:
                    replacements[atom] = false if lin.coefficient == 1 else true
        disjuncts.append(_substitute(normalized, replacements))
    # a solution which is not one of a whole descending chain of solutions
    # with step delta lies within delta above a threshold
    for bound in thresholds:
        for j in range(1, delta + 1):
            disjuncts.append(as_boolean(normalized.subs(x, as_expr(bound + j))))
    return _clean(as_boolean(Or(*disjuncts)))


def _point(disjunct: Boolean) -> Optional[tuple[Symbol, Expr]]:
    """``(x, e)`` when the disjunct *pins* ``x`` to ``e``: the equation
    ``x = e`` (``e`` free of ``x``) is the disjunct itself or one of its
    top-level conjuncts.

    An equation nested deeper -- inside an ``Or`` within the disjunct --
    does not pin anything: ``(m >= 0 | m = 0) & (m = 0 | 3 | m)`` holds
    for every multiple of 3, not only at ``m = 0``, and reading it as a
    point made :func:`_drop_covered` discard it (sympy-extras#27).
    """
    conjuncts = list(disjunct.args) if isinstance(disjunct, And) else [disjunct]
    for atom in conjuncts:
        if not isinstance(atom, Eq) or isinstance(atom.lhs, Mod):
            continue
        e = as_expr(atom.lhs - atom.rhs)
        for x in sorted(free_symbols(e), key=lambda s: s.name):
            lin = _split_linear(e, x)
            if lin is not None and lin.coefficient in (1, -1):
                return x, as_expr(-lin.rest*lin.coefficient)
    return None


def _drop_covered(formula: Boolean) -> Boolean:
    """Drop the disjuncts which fix the value of a variable when the other
    disjuncts hold at that value (the test points just above a threshold
    are often covered by the periodic part)."""
    if not isinstance(formula, Or):
        return formula
    disjuncts = [as_boolean(a) for a in formula.args]
    kept: list[Boolean] = []
    for i, d in enumerate(disjuncts):
        point = _point(d)
        if point is not None:
            # a disjunct is redundant only when the disjuncts that will
            # actually remain cover its point: the ones kept so far and
            # the ones not examined yet. Testing against the original
            # list let two disjuncts pinning the same point cover each
            # other, and both were dropped (sympy-extras#45, #51).
            others = Or(*kept, *disjuncts[i + 1:])
            if as_boolean(others.subs(point[0], point[1])) is true:
                continue
        kept.append(d)
    return as_boolean(Or(*kept))


def _clean(formula: Boolean) -> Boolean:
    from sympy.logic.boolalg import simplify_logic
    formula = _canonical_divisibilities(formula)
    if isinstance(formula, (BooleanTrue, BooleanFalse)):
        return formula
    try:
        result = as_boolean(simplify_logic(formula, deep=False))
    except (ValueError, TypeError):
        result = formula
    from sympy_extras.polys.virtual_substitution import with_sides
    return with_sides(_drop_covered(result))


def presburger_quantifier_elimination(formula: Boolean, prefix: QuantifierPrefix) -> Boolean:
    """Eliminate all the quantifiers of a formula of Presburger
    arithmetic (innermost first; a universal quantifier through
    `\\neg \\exists \\neg`).

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.integers import presburger_quantifier_elimination
    >>> presburger_quantifier_elimination((2*x > y) & (2*x < y + 2), [('exists', x)])
    Eq(Mod(y + 1, 2), 0)
    >>> presburger_quantifier_elimination(x < y, [('exists', y), ('forall', x)])
    False
    """
    current = as_boolean(formula)
    for kind, v in reversed(list(prefix)):
        if kind == 'exists':
            current = cooper(current, v)
        else:
            current = _clean(Not(cooper(Not(current), v)))
    return current
