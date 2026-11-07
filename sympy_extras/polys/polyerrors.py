"""The exceptions of the polynomial layer, SymPy's names and hierarchy.

Ported from SymPy's ``sympy.polys.polyerrors``. The exceptions carry a
message (SymPy's constructors with operands, ``ExactQuotientFailed(f, g,
dom)`` and the like, take the message here: the operands are formatted
by the caller).

>>> from sympy_extras.polys.polyerrors import PolynomialError, BasePolynomialError, GeneratorsNeeded
>>> issubclass(GeneratorsNeeded, BasePolynomialError), str(PolynomialError('x is not a polynomial'))
(True, 'x is not a polynomial')
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/polyerrors.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: every exception takes its message (no operand constructors).
# ---------------------------------------------------------------------

from __future__ import annotations

__all__ = ['BasePolynomialError', 'ExactQuotientFailed', 'PolynomialDivisionFailed', 'OperationNotSupported',
           'HeuristicGCDFailed', 'ModularGCDFailed', 'HomomorphismFailed', 'IsomorphismFailed', 'ExtraneousFactors',
           'EvaluationFailed', 'RefinementFailed', 'CoercionFailed', 'NotInvertible', 'NotReversible', 'NotAlgebraic',
           'DomainError', 'PolynomialError', 'UnificationFailed', 'UnsolvableFactorError', 'GeneratorsError',
           'GeneratorsNeeded', 'ComputationFailed', 'UnivariatePolynomialError', 'MultivariatePolynomialError',
           'PolificationFailed', 'OptionError', 'FlagError']


class BasePolynomialError(Exception):
    """The base class of the polynomial exceptions."""


class ExactQuotientFailed(BasePolynomialError):
    """An exact division whose divisor does not divide."""


class PolynomialDivisionFailed(BasePolynomialError):
    """A division which cannot proceed (a leading coefficient not
    invertible)."""


class OperationNotSupported(BasePolynomialError):
    """An operation the representation of a polynomial does not support."""


class HeuristicGCDFailed(BasePolynomialError):
    pass


class ModularGCDFailed(BasePolynomialError):
    pass


class HomomorphismFailed(BasePolynomialError):
    pass


class IsomorphismFailed(BasePolynomialError):
    pass


class ExtraneousFactors(BasePolynomialError):
    pass


class EvaluationFailed(BasePolynomialError):
    pass


class RefinementFailed(BasePolynomialError):
    pass


class CoercionFailed(BasePolynomialError):
    """An element which cannot be converted into a domain."""


class NotInvertible(BasePolynomialError):
    pass


class NotReversible(BasePolynomialError):
    pass


class NotAlgebraic(BasePolynomialError):
    pass


class DomainError(BasePolynomialError):
    pass


class PolynomialError(BasePolynomialError):
    """An expression which is not a polynomial in the generators."""


class UnificationFailed(BasePolynomialError):
    pass


class UnsolvableFactorError(BasePolynomialError):
    pass


class GeneratorsError(BasePolynomialError):
    pass


class GeneratorsNeeded(GeneratorsError):
    pass


class ComputationFailed(BasePolynomialError):
    pass


class UnivariatePolynomialError(PolynomialError):
    pass


class MultivariatePolynomialError(PolynomialError):
    pass


class PolificationFailed(PolynomialError):
    pass


class OptionError(BasePolynomialError):
    pass


class FlagError(OptionError):
    pass
