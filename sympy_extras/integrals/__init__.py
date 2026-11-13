"""Definite and indefinite integration extending :mod:`sympy.integrals`.

The package exports nothing: the modules are imported by their paths
(``from sympy_extras.integrals.definite import definite_integral``), so
that importing the native ones (:mod:`~sympy_extras.integrals.conditions`)
does not load the modules still written on SymPy. The integrators which
are written on SymPy take the conditions from the face
:mod:`~sympy_extras.integrals.conditions_sympy`."""
from __future__ import annotations

__all__: list[str] = []
