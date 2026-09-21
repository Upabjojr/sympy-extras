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
time_scale : float
    A factor applied to every time limit of the package (``timeout`` and
    the limits of the single steps alike): what the package finds within
    its limits depends on the speed of the machine, and a machine which is
    twice as slow gives the same answers with ``time_scale = 2``. It is
    read from the environment variable ``SYMPY_EXTRAS_TIME_SCALE`` when the
    package is imported (1 without it); the continuous integration, whose
    machines are slower than those the limits were chosen on, sets it.
modular_groebner : bool
    Whether the Gröbner bases of :class:`~sympy_extras.polys.ideals.Ideal`
    over the rationals are computed by the modular algorithm
    (:mod:`sympy_extras.polys.modulargroebner`) when SymPy's direct
    computation does not finish within ``groebner_direct_time``. The
    results are the same, reduced and proven, either way.
groebner_direct_time : float
    The seconds given to SymPy's direct computation of a Gröbner basis
    before the modular algorithm takes over (0.25, and twenty times as
    long for the graded orders, where the modular algorithm seldom pays
    off); zero or less to go to the modular algorithm at once.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

__all__ = ['Settings', 'settings', 'configure']


def _time_scale_of_the_environment() -> float:
    """The value of ``SYMPY_EXTRAS_TIME_SCALE``: 1 when it is not set, or
    not a positive number."""
    given = os.environ.get('SYMPY_EXTRAS_TIME_SCALE', '')
    try:
        scale = float(given)
    except ValueError:
        return 1.0
    return scale if 0 < scale < float('inf') else 1.0


class Settings:
    """The mutable global settings, see the module documentation."""

    def __init__(self) -> None:
        self.numerical_checks: bool = True
        self.precision: int = 30
        self.timeout: Optional[float] = 30.0
        self.time_scale: float = _time_scale_of_the_environment()
        self.modular_groebner: bool = True
        self.groebner_direct_time: float = 0.25

    def __repr__(self) -> str:
        return "Settings(numerical_checks=%r, precision=%r, timeout=%r)" % (
            self.numerical_checks, self.precision, self.timeout)


settings = Settings()


@contextmanager
def configure(numerical_checks: Optional[bool] = None, precision: Optional[int] = None,
              timeout: Optional[float] = None, no_timeout: bool = False,
              modular_groebner: Optional[bool] = None,
              groebner_direct_time: Optional[float] = None,
              time_scale: Optional[float] = None) -> Iterator[Settings]:
    """Temporarily change the settings (``no_timeout=True`` removes the
    time limit, since ``timeout=None`` means "leave unchanged" here)."""
    previous = (settings.numerical_checks, settings.precision, settings.timeout)
    previous_groebner = (settings.modular_groebner, settings.groebner_direct_time)
    previous_scale = settings.time_scale
    if time_scale is not None:
        if not time_scale > 0:
            raise ValueError("time_scale must be positive")
        settings.time_scale = time_scale
    if numerical_checks is not None:
        settings.numerical_checks = numerical_checks
    if precision is not None:
        settings.precision = precision
    if timeout is not None:
        settings.timeout = timeout
    if no_timeout:
        settings.timeout = None
    if modular_groebner is not None:
        settings.modular_groebner = modular_groebner
    if groebner_direct_time is not None:
        settings.groebner_direct_time = groebner_direct_time
    try:
        yield settings
    finally:
        settings.numerical_checks, settings.precision, settings.timeout = previous
        settings.modular_groebner, settings.groebner_direct_time = previous_groebner
        settings.time_scale = previous_scale
