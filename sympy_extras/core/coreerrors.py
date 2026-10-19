"""The exceptions of the core."""
from __future__ import annotations

__all__ = ['BaseCoreError', 'NonCommutativeExpression', 'PoleError', 'PrecisionExhausted']


class BaseCoreError(Exception):
    """The base class of the errors of the core."""


class NonCommutativeExpression(BaseCoreError):
    """An operation which needs commutative factors met noncommutative ones."""


class PoleError(BaseCoreError):
    """A series expansion was asked at a pole it cannot handle."""


class PrecisionExhausted(ArithmeticError):
    """A numerical evaluation could not reach the precision asked for."""
