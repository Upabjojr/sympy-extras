"""Regular chains and triangular decompositions of polynomial systems
(native: on the objects of the core, no SymPy; the SymPy face is
:mod:`sympy_extras.polys.regularchains_sympy`)."""
from .regularchain import RegularChain, regular_gcd, triangularize

__all__ = ['RegularChain', 'triangularize', 'regular_gcd']
