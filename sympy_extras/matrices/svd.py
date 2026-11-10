r"""Exact singular values and the singular value decomposition.

The singular values of a real matrix $A$ are the square roots of the
eigenvalues of the symmetric matrix $A^T A$, and a decomposition $A = U
\Sigma V^T$ is made of orthonormal eigenvectors $v_i$ of $A^T A$ and the
vectors $u_i = A v_i / \sigma_i$ [1]_. Here it is computed exactly:

* the eigenvectors of $A^T A$ are found once for all the roots of each
  irreducible factor $f$ of its characteristic polynomial, by elimination
  over $\mathbb{Q}[t]/(f)$ (:mod:`sympy_extras.matrices.eigen`);
* inside an eigenspace of dimension greater than one they are made
  orthogonal by the Gram–Schmidt process *without square roots*, which
  stays in that field (the squared norm of a real eigenvector is not
  zero at any root, so it is invertible there);
* the roots $\lambda$ of $f$ are substituted, $\sigma = \sqrt{\lambda}$,
  and each vector is divided by the square root of its squared norm. The
  three real roots of an irreducible cubic are written in Viete's
  trigonometric form, since their radical form goes through complex
  numbers.

Eigenvectors for different eigenvalues of a symmetric matrix are
orthogonal, so nothing else is needed. The decomposition returned is the
compact one: only the nonzero singular values, in decreasing order.

Examples
========

>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.matrices.svd import singular_values, svd
>>> A = ExprMatrix.from_rows([[3, 0], [4, 5]])
>>> singular_values(A)
[3*sqrt(5), sqrt(5)]
>>> U, S, V = svd(ExprMatrix.from_rows([[2, 0], [0, -3], [0, 0]]))
>>> U, S, V
(ExprMatrix([[0, 1], [-1, 0], [0, 0]]), ExprMatrix([[3, 0], [0, 2]]), ExprMatrix([[0, 1], [1, 0]]))
>>> U*S*V.transpose() == ExprMatrix.from_rows([[2, 0], [0, -3], [0, 0]])
True

References
==========

.. [1] G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
       Hopkins University Press, 2013, Section 2.4 (the singular value
       decomposition) and Section 8.6.
.. [2] R. A. Horn, C. R. Johnson, *Matrix Analysis*, 2nd ed., Cambridge
       University Press, 2013, Section 2.6.
"""
from __future__ import annotations

from typing import Sequence, TypeVar

from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Integer, Rational, number_from_fraction, pi
from sympy_extras.core.power import sqrt
from sympy_extras.functions import acos, cos
from sympy_extras.polys.domains.quotientfield import QuotientField
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly

from .dense import ExprMatrix
from .eigen import (_Field, _RationalField, _characteristic_factors, _column, _extension, _field_nullspace, _fresh_symbol,
                    _shifted, eigenvalues)

__all__ = ['singular_values', 'svd']

T = TypeVar('T')


def _gram(M: ExprMatrix) -> ExprMatrix:
    return M.transpose() * M


def singular_values(M: ExprMatrix) -> list[Expr]:
    """The singular values with their multiplicities (zeros included, as
    many as columns), the symbols taken to be real: the square roots of
    the eigenvalues of ``M.T * M``. In decreasing order for a matrix of
    numbers.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.svd import singular_values
    >>> a = Symbol('a')
    >>> singular_values(ExprMatrix.from_rows([[1, 1], [1, 1]])), singular_values(ExprMatrix.from_rows([[a, 0], [0, 2]]))
    ([2, 0], [2, sqrt(a**2)])
    """
    values: list[Expr] = []
    for value, multiplicity in eigenvalues(_gram(M)).items():
        values.extend([sqrt(value)] * multiplicity)
    if all(not v.free_symbols for v in values):
        values.sort(key=float, reverse=True)
    return values


def _orthogonal(F: QuotientField[T], vectors: Sequence[list[list[T]]]) -> list[tuple[list[list[T]], list[T]]]:
    """Gram-Schmidt without normalisation over ``F``: the orthogonal
    vectors with their squared norms."""
    def dot(u: Sequence[list[T]], v: Sequence[list[T]]) -> list[T]:
        total = F.zero
        for a, b in zip(u, v):
            if a and b:
                total = F.add(total, F.mul(a, b))
        return total

    done: list[tuple[list[list[T]], list[T]]] = []
    for vector in vectors:
        current = list(vector)
        for other, norm in done:
            factor = F.mul(dot(current, other), F.invert(norm))
            if factor:
                current = [F.sub(a, F.mul(factor, b)) for a, b in zip(current, other)]
        done.append((current, dot(current, current)))
    return done


def _real_roots(factor: Expr, unknown: Expr) -> list[Expr]:
    """The roots of an irreducible factor of the characteristic
    polynomial of a symmetric matrix of rational numbers, which are real:
    a cubic in the trigonometric form of Viete (the radical form goes
    through complex numbers), the others in radicals."""
    polynomial = Poly(factor, unknown)
    coefficients = polynomial.all_coeffs()
    if polynomial.degree() == 3 and all(isinstance(c, Rational) for c in coefficients):
        a, b, c, d = (v.as_fraction() for v in coefficients if isinstance(v, Rational))
        # x = y - b/(3a): y**3 + p*y + q with p < 0 for three real roots
        p = (3*a*c - b*b)/(3*a*a)
        q = (2*b**3 - 9*a*b*c + 27*a*a*d)/(27*a**3)
        shift = number_from_fraction(-b/(3*a))
        radius = 2*sqrt(number_from_fraction(-p/3))
        angle = acos(number_from_fraction(3*q/(2*p))*sqrt(number_from_fraction(-3/p)))/3
        return [shift + radius*cos(angle - 2*pi*k/3) for k in range(3)]
    found = roots(factor, unknown)
    if not isinstance(found, dict) or sum(found.values()) != polynomial.degree():
        raise NotImplementedError('the roots of %s are not expressed in radicals' % (factor,))
    return list(found)


def _singular_vectors(field: _Field[T], gram: ExprMatrix) -> list[tuple[Expr, ExprMatrix]]:
    """``(lambda, v)``: the positive eigenvalues of the Gram matrix with
    orthonormal eigenvectors."""
    unknown = _fresh_symbol(gram)
    _, factors = _characteristic_factors(gram, unknown)
    zero: Expr = Integer(0)
    pairs: list[tuple[Expr, ExprMatrix]] = []
    for factor, _ in factors:
        if factor.subs(unknown, zero) == 0:
            continue
        F = _extension(field, factor, unknown)
        orthogonal = _orthogonal(F, _field_nullspace(_shifted(field, F, gram)))
        for value in _real_roots(factor, unknown):
            for vector, norm in orthogonal:
                column = _column(field, vector, unknown).subs(unknown, value)
                length = sqrt(_column(field, [norm], unknown)[0, 0].subs(unknown, value))
                pairs.append((value, column / length))
    return pairs


def svd(M: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix, ExprMatrix]:
    """The compact singular value decomposition ``(U, S, V)`` of a matrix
    of rational numbers of rank ``r``: ``M = U * S * V.T`` with ``S`` the
    ``r`` by ``r`` diagonal matrix of the nonzero singular values in
    decreasing order, and ``U`` (``m`` by ``r``) and ``V`` (``n`` by
    ``r``) with orthonormal columns.

    ``NotImplementedError`` when the singular values are not expressed in
    radicals.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.svd import svd
    >>> U, S, V = svd(ExprMatrix.from_rows([[1, 1], [1, 1]]))
    >>> U, S, V
    (ExprMatrix([[sqrt(2)/2], [sqrt(2)/2]]), ExprMatrix([[2]]), ExprMatrix([[sqrt(2)/2], [sqrt(2)/2]]))
    """
    if not all(isinstance(e, Rational) for e in M.entries):
        raise NotImplementedError('the singular value decomposition of a matrix whose entries are not rational numbers')
    pairs = _singular_vectors(_RationalField(), _gram(M))
    pairs.sort(key=lambda pair: float(pair[0]), reverse=True)
    r = len(pairs)
    zero: Expr = Integer(0)
    sigma = [sqrt(value) for value, _ in pairs]
    left = [(M * vector) / s for (_, vector), s in zip(pairs, sigma)]
    U = ExprMatrix(M.rows, r, [left[j][i, 0] for i in range(M.rows) for j in range(r)])
    V = ExprMatrix(M.cols, r, [pairs[j][1][i, 0] for i in range(M.cols) for j in range(r)])
    S = ExprMatrix(r, r, [sigma[i] if i == j else zero for i in range(r) for j in range(r)])
    return U, S, V
