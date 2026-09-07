"""Global settings of sympy-extras.

``settings`` is the single instance; change its attributes directly or
temporarily with :func:`configure`::

    >>> from sympy_extras.settings import settings, configure
    >>> settings.numerical_checks
    True
    >>> with configure(numerical_checks=False):
    ...     settings.numerical_checks
    False
    >>> settings.numerical_checks
    True

Attributes
==========

numerical_checks : bool
    Whether the assumptions module may use certified numerical evaluation
    (with the precision below and a check of the error bound) to decide
    the sign of constant expressions such as ``sin(3) - 1/7`` and to
    discard the candidates returned by SymPy's solvers which do not
    satisfy their equation. Off, only exact evaluation is used: slower
    answers become ``None`` and some extraneous solutions are kept.
precision : int
    The number of significant digits of the numerical checks (30).
timeout : float or None
    The default time limit, in seconds, of every step handed to SymPy by
    the solvers (``dsolve``, ``integrate``, ``solve``, the verifications);
    ``None`` for no limit.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

__all__ = ['Settings', 'settings', 'configure']


class Settings:
    """The mutable global settings, see the module documentation."""

    def __init__(self) -> None:
        self.numerical_checks: bool = True
        self.precision: int = 30
        self.timeout: Optional[float] = 30.0

    def __repr__(self) -> str:
        return "Settings(numerical_checks=%r, precision=%r, timeout=%r)" % (
            self.numerical_checks, self.precision, self.timeout)


settings = Settings()


@contextmanager
def configure(numerical_checks: Optional[bool] = None, precision: Optional[int] = None,
              timeout: Optional[float] = None, no_timeout: bool = False) -> Iterator[Settings]:
    """Temporarily change the settings (``no_timeout=True`` removes the
    time limit, since ``timeout=None`` means "leave unchanged" here)."""
    previous = (settings.numerical_checks, settings.precision, settings.timeout)
    if numerical_checks is not None:
        settings.numerical_checks = numerical_checks
    if precision is not None:
        settings.precision = precision
    if timeout is not None:
        settings.timeout = timeout
    if no_timeout:
        settings.timeout = None
    try:
        yield settings
    finally:
        settings.numerical_checks, settings.precision, settings.timeout = previous
