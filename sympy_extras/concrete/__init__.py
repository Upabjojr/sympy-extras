"""Summation algorithms extending :mod:`sympy.concrete`.

The package exports nothing: the modules are imported by their paths
(``from sympy_extras.concrete.karr import karr_sum``), so that importing
the native ones (:mod:`~sympy_extras.concrete.expr_with_limits`, used by
the core) does not load the modules still written on SymPy.
"""
from __future__ import annotations

__all__: list[str] = []
