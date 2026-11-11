r"""Linear systems in state space form, exactly: controllability and
observability, the transfer function, pole placement and realisations.

For the system $x' = A x + B u$, $y = C x + D u$ (or its discrete
counterpart) with matrices of expressions:

* :func:`controllability_matrix` is $(B, A B, \dots, A^{n-1} B)$ and
  :func:`observability_matrix` its dual; the system is controllable
  (observable) exactly when the rank is $n$ (Kalman's criterion [1]_),
  for generic values of the symbols;
* :func:`transfer_function` is $G(s) = C (s I - A)^{-1} B + D$, computed
  by the fraction-free elimination of
  :mod:`sympy_extras.matrices.linalg` over the polynomials in $s$ and the
  parameters, each entry in lowest terms;
* :func:`pole_placement` gives, for a single input, the feedback $u = -K
  x$ for which $A - B K$ has a prescribed characteristic polynomial $p$,
  by Ackermann's formula [2]_

  .. math:: K = (0, \dots, 0, 1)\, (B, A B, \dots, A^{n-1} B)^{-1}\, p(A);

* :func:`controllable_decomposition` and :func:`observable_decomposition`
  are Kalman's decompositions: a change of state which separates the
  part the inputs reach (the outputs see) from the rest;
  :func:`minimal_realization` removes both remainders;
* :func:`realization` writes a proper rational function $G(s)$ as a
  system in controllable canonical form [1]_, of the order of its
  denominator (minimal when the fraction is in lowest terms).

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.matrices.statespace import (controllability_matrix, is_controllable, transfer_function,
...     pole_placement, realization)
>>> s, k, m = Symbol('s'), Symbol('k'), Symbol('m')
>>> A, B = ExprMatrix.from_rows([[0, 1], [-k/m, 0]]), ExprMatrix.column([0, 1/m])
>>> controllability_matrix(A, B), is_controllable(A, B)
(ExprMatrix([[0, 1/m], [1/m, 0]]), True)
>>> transfer_function(A, B, ExprMatrix.from_rows([[1, 0]]), s)
ExprMatrix([[1/(k + m*s**2)]])
>>> pole_placement(A, B, (s + 1)*(s + 2), s)
ExprMatrix([[-k + 2*m, 3*m]])
>>> realization((s + 3)/(s**2 + 3*s + 2), s)
(ExprMatrix([[0, 1], [-2, -3]]), ExprMatrix([[0], [1]]), ExprMatrix([[3, 1]]), ExprMatrix([[0]]))

References
==========

.. [1] T. Kailath, *Linear Systems*, Prentice-Hall, 1980, Sections 2.1
       (realisations, canonical forms), 2.3 and 2.4 (controllability and
       observability).
.. [2] J. Ackermann, *Der Entwurf linearer Regelungssysteme im
       Zustandsraum*, Regelungstechnik 20 (1972), 297-300.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence, Union

from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.numbers import Integer
from sympy_extras.core.symbol import Symbol
from sympy_extras.simplify.radsimp import fraction

from .dense import ExprMatrix
from .eigen import _coefficients
from .linalg import _square, inverse, normal, rank, rref

__all__ = ['controllability_matrix', 'observability_matrix', 'is_controllable', 'is_observable', 'transfer_function',
           'pole_placement', 'realization', 'controllable_decomposition', 'observable_decomposition', 'minimal_realization']


def _hstack(blocks: Sequence[ExprMatrix]) -> ExprMatrix:
    rows = blocks[0].rows
    return ExprMatrix(rows, sum(b.cols for b in blocks), [e for i in range(rows) for b in blocks for e in b.row(i)])


def controllability_matrix(A: ExprMatrix, B: ExprMatrix) -> ExprMatrix:
    """``(B, A B, ..., A**(n-1) B)``.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.statespace import controllability_matrix
    >>> controllability_matrix(ExprMatrix.from_rows([[1, 1], [0, 2]]), ExprMatrix.column([0, 1]))
    ExprMatrix([[0, 1], [1, 2]])
    """
    n = _square(A)
    if B.rows != n:
        raise ValueError('B has %d rows for a state of dimension %d' % (B.rows, n))
    blocks = [B]
    for _ in range(n - 1):
        blocks.append((A*blocks[-1]).applyfunc(normal))
    return _hstack(blocks) if n else B


def observability_matrix(A: ExprMatrix, C: ExprMatrix) -> ExprMatrix:
    """The rows ``C, C A, ..., C A**(n-1)`` stacked."""
    n = _square(A)
    if C.cols != n:
        raise ValueError('C has %d columns for a state of dimension %d' % (C.cols, n))
    return controllability_matrix(A.transpose(), C.transpose()).transpose()


def is_controllable(A: ExprMatrix, B: ExprMatrix) -> bool:
    """Whether the pair ``(A, B)`` is controllable (for generic values of
    the symbols).

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.statespace import is_controllable
    >>> A = ExprMatrix.from_rows([[1, 0], [0, 2]])
    >>> is_controllable(A, ExprMatrix.column([1, 1])), is_controllable(A, ExprMatrix.column([1, 0]))
    (True, False)
    """
    return rank(controllability_matrix(A, B)) == A.rows


def is_observable(A: ExprMatrix, C: ExprMatrix) -> bool:
    """Whether the pair ``(A, C)`` is observable (for generic values of
    the symbols)."""
    return rank(observability_matrix(A, C)) == A.rows


def transfer_function(A: ExprMatrix, B: ExprMatrix, C: ExprMatrix, s: Symbol, D: Optional[ExprMatrix] = None) -> ExprMatrix:
    """The matrix ``C (s I - A)**-1 B + D`` of the transfer functions
    from the inputs to the outputs, each in lowest terms.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.statespace import transfer_function
    >>> s = Symbol('s')
    >>> transfer_function(ExprMatrix.from_rows([[0, 1], [-2, -3]]), ExprMatrix.column([0, 1]), ExprMatrix.from_rows([[1, 0]]), s)
    ExprMatrix([[1/(s**2 + 3*s + 2)]])
    """
    n = _square(A)
    if B.rows != n or C.cols != n:
        raise ValueError('shapes: A %s, B %s, C %s' % (A.shape, B.shape, C.shape))
    if any(e.has(s) for M in (A, B, C) for e in M.entries):
        raise ValueError('%s occurs in the matrices of the system' % (s,))
    if n == 0:
        result = ExprMatrix.zeros(C.rows, B.cols)
    else:
        resolvent = inverse(ExprMatrix.eye(n)*s - A)
        result = C*resolvent*B
    if D is not None:
        if D.shape != (C.rows, B.cols):
            raise ValueError('D has the shape %s for %d outputs and %d inputs' % (D.shape, C.rows, B.cols))
        result = result + D
    return result.applyfunc(normal)


def pole_placement(A: ExprMatrix, B: ExprMatrix, target: Union[ExprLike, Sequence[ExprLike]], s: Optional[Symbol] = None
                   ) -> ExprMatrix:
    """The gain ``K`` (one row) for which ``A - B K`` has the prescribed
    characteristic polynomial, for a system with one input: ``target`` is
    a monic polynomial of degree ``n`` in ``s``, or the list of the ``n``
    closed loop poles. ``ValueError`` when the pair is not controllable.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.statespace import pole_placement
    >>> A, B = ExprMatrix.from_rows([[0, 1], [0, 0]]), ExprMatrix.column([0, 1])
    >>> pole_placement(A, B, [-1, -2])
    ExprMatrix([[2, 3]])
    """
    n = _square(A)
    if B.shape != (n, 1):
        raise ValueError('pole placement by the formula of Ackermann is for one input: B has the shape %s' % (B.shape,))
    coefficients: list[Expr]
    if isinstance(target, (Expr, int, float, Fraction)):
        if s is None:
            raise ValueError('the variable of the polynomial is needed')
        polynomial = expand(to_expr(target))
        coefficients = _coefficients(polynomial, s)
        if coefficients[0] != 1:
            coefficients = [normal(c/coefficients[0]) for c in coefficients]
    else:
        coefficients = [Integer(1)]
        for pole in target:
            root = to_expr(pole)
            shifted = coefficients + [Integer(0)]
            coefficients = [shifted[i] - (root*coefficients[i - 1] if i else 0) for i in range(len(shifted))]
        coefficients = [normal(c) for c in coefficients]
    if len(coefficients) != n + 1:
        raise ValueError('a polynomial of degree %d is needed' % n)
    if n == 0:
        return ExprMatrix.zeros(1, 0)
    controllability = controllability_matrix(A, B)
    try:
        inverted = inverse(controllability)
    except ValueError:
        raise ValueError('the pair (A, B) is not controllable') from None
    # p(A) by Horner's rule
    value = ExprMatrix.eye(n)*coefficients[0]
    for c in coefficients[1:]:
        value = (value*A + ExprMatrix.eye(n)*c).applyfunc(normal)
    last = ExprMatrix(1, n, [inverted[n - 1, j] for j in range(n)])
    return (last*value).applyfunc(normal)


def realization(G: ExprLike, s: Symbol) -> tuple[ExprMatrix, ExprMatrix, ExprMatrix, ExprMatrix]:
    """``(A, B, C, D)`` in controllable canonical form with the transfer
    function ``G``, a proper rational function of ``s`` (the degree of
    the numerator at most that of the denominator).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.statespace import realization
    >>> s = Symbol('s')
    >>> realization(2*s/(s + 1), s)
    (ExprMatrix([[-1]]), ExprMatrix([[1]]), ExprMatrix([[-2]]), ExprMatrix([[2]]))
    """
    numerator, denominator = fraction(normal(to_expr(G)))
    top = [Integer(0)] if expand(numerator) == 0 else _coefficients(expand(numerator), s)
    bottom = _coefficients(expand(denominator), s)
    n = len(bottom) - 1
    if len(top) - 1 > n:
        raise ValueError('%s is not proper: the degree of the numerator exceeds that of the denominator' % (G,))
    leading = bottom[0]
    a = [normal(c/leading) for c in bottom]                       # monic denominator s**n + a[1] s**(n-1) + ...
    zero: Expr = Integer(0)
    b = [zero]*(n + 1 - len(top)) + [normal(c/leading) for c in top]
    direct = b[0]
    # the strictly proper part: (b[k] - direct*a[k]) s**(n-k), k = 1..n
    rest = [normal(b[k] - direct*a[k]) for k in range(1, n + 1)]
    A = ExprMatrix(n, n, [(-a[n - j] if i == n - 1 else (Integer(1) if j == i + 1 else zero))
                          for i in range(n) for j in range(n)])
    B = ExprMatrix(n, 1, [Integer(1) if i == n - 1 else zero for i in range(n)])
    C = ExprMatrix(1, n, [rest[n - 1 - j] for j in range(n)])
    return A, B, C, ExprMatrix(1, 1, [direct])




def _completed_basis(M: ExprMatrix) -> tuple[ExprMatrix, int]:
    """A square invertible matrix whose first ``r`` columns are a basis
    of the column space of ``M`` (its pivot columns), completed by
    vectors of the standard basis; and ``r``."""
    n = M.rows
    extended = _hstack([M, ExprMatrix.eye(n)]) if M.cols else ExprMatrix.eye(n)
    _, pivots = rref(extended)
    r = sum(1 for column in pivots if column < M.cols)
    return ExprMatrix(n, n, [extended[i, j] for i in range(n) for j in pivots]), r


def controllable_decomposition(A: ExprMatrix, B: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix, ExprMatrix, int]:
    """``(T, A1, B1, r)``: a change of state ``x = T z`` after which the
    first ``r`` coordinates are the controllable part,

    ``A1 = T**-1 A T = [[Ac, *], [0, Au]]``, ``B1 = T**-1 B = [[Bc], [0]]``

    with ``(Ac, Bc)`` controllable of order ``r`` (Kalman's
    decomposition; the eigenvalues of ``Au`` are the modes which no input
    reaches).

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.statespace import controllable_decomposition
    >>> T, A1, B1, r = controllable_decomposition(ExprMatrix.from_rows([[1, 1], [0, 2]]), ExprMatrix.column([1, 0]))
    >>> A1, B1, r
    (ExprMatrix([[1, 1], [0, 2]]), ExprMatrix([[1], [0]]), 1)
    """
    T, r = _completed_basis(controllability_matrix(A, B))
    inverted = inverse(T)
    return T, (inverted*A*T).applyfunc(normal), (inverted*B).applyfunc(normal), r


def observable_decomposition(A: ExprMatrix, C: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix, ExprMatrix, int]:
    """``(T, A1, C1, r)``: a change of state after which the first ``r``
    coordinates are the observable part,

    ``A1 = T**-1 A T = [[Ao, 0], [*, Au]]``, ``C1 = C T = [Co, 0]``

    with ``(Ao, Co)`` observable of order ``r``."""
    S, _, _, r = controllable_decomposition(A.transpose(), C.transpose())
    # the dual change of basis: T = (S^T)**-1
    T = inverse(S.transpose())
    return T, (S.transpose()*A*T).applyfunc(normal), (C*T).applyfunc(normal), r


def minimal_realization(A: ExprMatrix, B: ExprMatrix, C: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix, ExprMatrix]:
    """A controllable and observable system ``(A, B, C)`` with the same
    transfer function ``C (s I - A)**-1 B``: the uncontrollable part is
    removed, then the unobservable part of what remains.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.statespace import minimal_realization
    >>> A = ExprMatrix.from_rows([[-1, 0], [0, -2]])
    >>> minimal_realization(A, ExprMatrix.column([1, 1]), ExprMatrix.from_rows([[1, 0]]))        # 1/(s + 1)
    (ExprMatrix([[-1]]), ExprMatrix([[1]]), ExprMatrix([[1]]))
    """
    if C.cols != A.rows:
        raise ValueError('C has %d columns for a state of dimension %d' % (C.cols, A.rows))
    T, A1, B1, r = controllable_decomposition(A, B)
    C1 = (C*T).applyfunc(normal)
    Ac = ExprMatrix(r, r, [A1[i, j] for i in range(r) for j in range(r)])
    Bc = ExprMatrix(r, B.cols, [B1[i, j] for i in range(r) for j in range(B.cols)])
    Cc = ExprMatrix(C.rows, r, [C1[i, j] for i in range(C.rows) for j in range(r)])
    _, A2, C2, q = observable_decomposition(Ac, Cc)
    S, _, _, _ = controllable_decomposition(Ac.transpose(), Cc.transpose())
    B2 = (S.transpose()*Bc).applyfunc(normal)
    return (ExprMatrix(q, q, [A2[i, j] for i in range(q) for j in range(q)]),
            ExprMatrix(q, B.cols, [B2[i, j] for i in range(q) for j in range(B.cols)]),
            ExprMatrix(C.rows, q, [C2[i, j] for i in range(C.rows) for j in range(q)]))
