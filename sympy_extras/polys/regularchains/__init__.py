"""Regular chains and triangular decompositions of polynomial systems."""
from .regularchain import RegularChain, regular_gcd, triangularize

__all__ = ['RegularChain', 'triangularize', 'regular_gcd']
