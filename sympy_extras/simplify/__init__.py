"""Simplification algorithms extending :mod:`sympy.simplify` (the ports of
its modules and the structure theorem, :mod:`~sympy_extras.simplify.structure`,
native, with its SymPy face :mod:`~sympy_extras.simplify.structure_sympy`).

The package exports nothing: the modules are imported by their paths
(``from sympy_extras.simplify.structure_sympy import canonical_form``), so that
importing the native ones (used by the core) does not load the modules
still written on SymPy.
"""
from __future__ import annotations

__all__: list[str] = []
