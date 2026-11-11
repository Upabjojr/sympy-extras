r"""The generalised eigenvalue problem $A v = \lambda B v$ for a pencil of
square matrices of expressions.

The finite eigenvalues of the pencil $(A, B)$ are the roots of
$\det(A - \lambda B)$, a polynomial of degree at most $n$; when $B$ is
singular the degree is lower and the pencil has the eigenvalue
$\infty$ with the missing multiplicity (the eigenvalue zero of $(B,
A)$). A pencil whose determinant vanishes identically is *singular* and
has no spectrum in this sense [1]_.

As for ordinary eigenvalues (:mod:`sympy_extras.matrices.eigen`), the
polynomial is factored over the field $K$ of the rational functions in
the symbols, and for each irreducible factor $f$ the null space of $A -
\lambda B$ is computed once, by elimination over $K[\lambda]/(f)$: the
eigenvectors are polynomials in the eigenvalue, and $B$ is never
inverted. The entries are rational functions of symbols.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.matrices.pencils import (pencil_polynomial, generalized_eigenvalues, generalized_eigenvectors,
...     generalized_eigenspaces)
>>> x, k, m = Symbol('x'), Symbol('k'), Symbol('m')
>>> K = ExprMatrix.from_rows([[2*k, -k], [-k, k]])                   # two masses on springs: K v = w**2 M v
>>> M = ExprMatrix.from_rows([[m, 0], [0, m]])
>>> pencil_polynomial(K, M, x)
k**2 - 3*k*m*x + m**2*x**2
>>> generalized_eigenspaces(K, M, x)
[Eigenspace(k**2 - 3*k*m*x + m**2*x**2, 1, [ExprMatrix([[1 - m*x/k], [1]])])]
>>> generalized_eigenvalues(ExprMatrix.eye(2), ExprMatrix.from_rows([[1, 0], [0, 0]]))
{1: 1, oo: 1}
>>> generalized_eigenvectors(ExprMatrix.from_rows([[2, 0], [0, 3]]), ExprMatrix.from_rows([[1, 0], [0, 1]]))
[(2, 1, [ExprMatrix([[1], [0]])]), (3, 1, [ExprMatrix([[0], [1]])])]

References
==========

.. [1] F. R. Gantmacher, *The Theory of Matrices*, vol. 2, Chelsea, 1959,
       Chapter XII (pencils of matrices: regular and singular pencils).
.. [2] G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
       Hopkins University Press, 2013, Section 7.7 (the generalised
       eigenvalue problem).
"""
from __future__ import annotations

from typing import Sequence, TypeVar

from sympy_extras.core.expr import Expr
from sympy_extras.core.function import expand
from sympy_extras.core.numbers import Integer, oo
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.domains.quotientfield import QuotientField
from sympy_extras.polys.polytools import factor_list

from .dense import ExprMatrix
from .domainmatrix import DomainMatrix
from .eigen import (Eigenspace, _Field, _FunctionField, _RationalField, _coefficients, _column, _extension,
                    _field_nullspace, _roots_in_radicals)
from .linalg import _PolynomialLift, _kernels, _square, det, normal

__all__ = ['pencil_polynomial', 'generalized_eigenspaces', 'generalized_eigenvalues', 'generalized_eigenvectors']

T = TypeVar('T')


def _check(A: ExprMatrix, B: ExprMatrix) -> int:
    n = _square(A)
    if _square(B) != n:
        raise ValueError('the matrices of a pencil have the same size: %s and %s' % (A.shape, B.shape))
    return n


def _fresh(A: ExprMatrix, B: ExprMatrix) -> Symbol:
    used = {s.name for M in (A, B) for e in M.entries for s in e.free_symbols if isinstance(s, Symbol)}
    name, k = '_lambda', 0
    while name in used:
        name, k = '_lambda%d' % k, k + 1
    return Symbol(name)


def pencil_polynomial(A: ExprMatrix, B: ExprMatrix, x: Expr) -> Expr:
    r"""The numerator of $\det(A - x B)$, expanded: its roots are the
    finite eigenvalues of the pencil. Zero for a singular pencil.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.pencils import pencil_polynomial
    >>> x = Symbol('x')
    >>> pencil_polynomial(ExprMatrix.from_rows([[1, 2], [3, 4]]), ExprMatrix.from_rows([[1, 0], [0, 0]]), x)
    -4*x - 2
    """
    n = _check(A, B)
    unknown = _fresh(A, B)
    determinant = det((A - B*unknown).applyfunc(normal)) if n else Integer(1)
    numerator = expand(normal(determinant).together().as_numer_denom()[0])
    return numerator.subs(unknown, x)


def _spaces(field: _Field[T], A: ExprMatrix, B: ExprMatrix, factors: Sequence[tuple[Expr, int]], unknown: Symbol, x: Symbol
            ) -> list[Eigenspace]:
    spaces: list[Eigenspace] = []
    for factor, multiplicity in factors:
        F: QuotientField[T] = _extension(field, factor, unknown)
        rows = [[F.sub(F.from_ground(field.element(p)), F.mul(F.generator, F.from_ground(field.element(q))))
                 for p, q in zip(A.row(i), B.row(i))] for i in range(A.rows)]
        basis = [_column(field, vector, x) for vector in _field_nullspace(DomainMatrix(rows, F, A.cols))]
        spaces.append(Eigenspace(factor.subs(unknown, x), x, multiplicity, basis))
    return spaces


def _factors(A: ExprMatrix, B: ExprMatrix, unknown: Symbol) -> tuple[list[Expr], list[tuple[Expr, int]], int]:
    """The kernels, the irreducible factors of the pencil polynomial
    with their multiplicities, and its degree."""
    kernels = _kernels([A, B])
    if not all(isinstance(k, Symbol) for k in kernels):
        raise NotImplementedError('pencils of matrices whose entries are not rational functions of symbols')
    polynomial = pencil_polynomial(A, B, unknown)
    if polynomial == 0:
        raise ValueError('the pencil is singular: det(A - x*B) vanishes identically')
    degree = len(_coefficients(polynomial, unknown)) - 1
    symbols = [k for k in kernels if isinstance(k, Symbol)]
    _, listed = factor_list(polynomial, unknown, *symbols)
    factors: list[tuple[Expr, int]] = []
    for factor, exponent in listed:
        if factor.has(unknown) and isinstance(exponent, Integer):
            factors.append((factor, exponent.p))
    return kernels, factors, degree


def generalized_eigenspaces(A: ExprMatrix, B: ExprMatrix, x: Symbol) -> list[Eigenspace]:
    """The finite eigenvalues and the eigenvectors of the pencil, grouped
    by the irreducible factors (in ``x``) of ``det(A - x*B)``: for each
    root ``x`` of the factor, the vectors of ``basis`` span the solutions
    of ``A v = x B v``. ``ValueError`` for a singular pencil.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.pencils import generalized_eigenspaces
    >>> x = Symbol('x')
    >>> generalized_eigenspaces(ExprMatrix.from_rows([[0, 1], [1, 0]]), ExprMatrix.from_rows([[1, 0], [0, 2]]), x)
    [Eigenspace(2*x**2 - 1, 1, [ExprMatrix([[2*x], [1]])])]
    """
    _check(A, B)
    unknown = _fresh(A, B)
    kernels, factors, _ = _factors(A, B, unknown)
    if kernels:
        return _spaces(_FunctionField(_PolynomialLift(kernels)), A, B, factors, unknown, x)
    return _spaces(_RationalField(), A, B, factors, unknown, x)


def _explicit(A: ExprMatrix, B: ExprMatrix) -> list[tuple[Expr, Eigenspace]]:
    pairs: list[tuple[Expr, Eigenspace]] = []
    for space in generalized_eigenspaces(A, B, _fresh(A, B)):
        values = _roots_in_radicals(space.factor, space.symbol)
        if values is None:
            raise NotImplementedError('the roots of %s are not expressed in radicals: see generalized_eigenspaces'
                                      % (space.factor,))
        pairs.extend((value, space) for value in values)
    pairs.sort(key=lambda pair: default_sort_key(pair[0]))
    return pairs


def generalized_eigenvalues(A: ExprMatrix, B: ExprMatrix) -> dict[Expr, int]:
    """The eigenvalues of the pencil with their algebraic
    multiplicities; ``oo`` has the multiplicity by which the degree of
    ``det(A - x*B)`` is lower than the size (a singular ``B``).

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.pencils import generalized_eigenvalues
    >>> generalized_eigenvalues(ExprMatrix.from_rows([[1, 2], [3, 4]]), ExprMatrix.from_rows([[1, 0], [0, 0]]))
    {-1/2: 1, oo: 1}
    """
    n = _check(A, B)
    _, _, degree = _factors(A, B, _fresh(A, B))
    result: dict[Expr, int] = {value: space.multiplicity for value, space in _explicit(A, B)}
    if degree < n:
        result[oo] = n - degree
    return result


def generalized_eigenvectors(A: ExprMatrix, B: ExprMatrix) -> list[tuple[Expr, int, list[ExprMatrix]]]:
    """``(eigenvalue, algebraic multiplicity, basis of the solutions of A
    v = eigenvalue B v)`` for every finite eigenvalue of the pencil."""
    return [(value, space.multiplicity, space.vectors(value)) for value, space in _explicit(A, B)]
