"""Matrices on the typed layers: :class:`ExprMatrix` (expressions of the
core, immutable) with its linear algebra
(:mod:`sympy_extras.matrices.linalg`), and :class:`DomainMatrix`
(elements of a typed domain, fraction-free algorithms)."""
from .dense import ExprMatrix
from .domainmatrix import DomainMatrix

__all__ = ['ExprMatrix', 'DomainMatrix']
