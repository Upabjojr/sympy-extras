"""Printers of the native objects: LaTeX, two-dimensional text, and the
support of interactive sessions.

``latex`` prints a node of the typed core, a matrix or a polynomial as
LaTeX, ``pretty`` as two-dimensional text in Unicode or ASCII (``pprint``
prints it); ``init_printing`` chooses how the results are shown in a
notebook or a Python session. The two printers share a typed framework
(:mod:`sympy_extras.printing.framework`): rules found by the class of the
node, which return what they printed with its binding strength, contexts
passed down by the parents, brackets decided in one place; and the
conventions of names (:mod:`sympy_extras.printing.conventions`).
"""
from .interactive import init_printing
from .latex import latex
from .pretty import pprint, pretty

__all__ = ['latex', 'pretty', 'pprint', 'init_printing']
