"""Simplification algorithms extending :mod:`sympy.simplify`: the canonical
form and the zero test of elementary expressions by the structure theorem
(:mod:`~sympy_extras.simplify.structure`)."""
from .structure import ElementaryTower, Element, Generator, NotElementary, canonical_form, is_zero, equal

__all__ = ['ElementaryTower', 'Element', 'Generator', 'NotElementary', 'canonical_form', 'is_zero', 'equal']
