"""Printers of the native objects: LaTeX, and the support of interactive
sessions.

``latex`` prints a node of the typed core, a matrix or a polynomial as
LaTeX; ``init_printing`` chooses how the results are shown in a notebook
or a Python session. The printers share a typed framework
(:mod:`sympy_extras.printing.framework`): rules found by the class of the
node, which return what they printed with its binding strength, contexts
passed down by the parents, brackets decided in one place.
"""
from .interactive import init_printing
from .latex import latex

__all__ = ['latex', 'init_printing']
