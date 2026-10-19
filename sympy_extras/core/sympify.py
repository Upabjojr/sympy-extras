"""Conversion of Python and SymPy objects to the expression tree.

``sympify`` accepts the nodes of this core, Python integers, floats,
fractions and booleans, strings (parsed by SymPy's parser and converted),
tuples and lists (which become :class:`~sympy_extras.core.containers.Tuple`),
dictionaries (:class:`~sympy_extras.core.containers.Dict`) and SymPy objects
(converted by :func:`~sympy_extras.core.interop.from_sympy`).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence, Union

from sympy.core.basic import Basic as SymPyBasic

from .basic import Basic

__all__ = ['sympify', 'SympifyError', 'SympifyLike']

#: what ``sympify`` accepts
SympifyLike = Union[Basic, int, float, Fraction, bool, str, SymPyBasic, Sequence['SympifyLike'],
    Mapping['SympifyLike', 'SympifyLike']]


class SympifyError(ValueError):
    """``sympify`` was given something it cannot convert."""


def sympify(a: SympifyLike) -> Basic:
    """``a`` as a node of the expression tree.

    Examples
    ========

    >>> from sympy_extras.core import sympify
    >>> from fractions import Fraction
    >>> sympify(2), sympify(Fraction(1, 2)), sympify(2.5), sympify(True)
    (2, 1/2, 2.50000000000000, True)
    >>> sympify('x**2 + 1')
    x**2 + 1
    >>> import sympy
    >>> sympify(sympy.Symbol('y')*2)
    2*y
    """
    from .containers import Tuple
    from .logic import false, true
    from .numbers import Float, Integer, number_from_fraction
    if isinstance(a, Basic):
        return a
    if isinstance(a, bool):
        return true if a else false
    if isinstance(a, int):
        return Integer(a)
    if isinstance(a, Fraction):
        return number_from_fraction(a)
    if isinstance(a, float):
        return Float(a)
    if isinstance(a, str):
        from sympy.parsing.sympy_parser import parse_expr
        from .interop import from_sympy
        from .parameters import global_parameters
        try:
            parsed = parse_expr(a, evaluate=global_parameters.evaluate)
        except (SyntaxError, ValueError, TypeError, NameError) as error:
            raise SympifyError("cannot parse %r: %s" % (a, error)) from None
        if not isinstance(parsed, SymPyBasic):
            raise SympifyError("cannot parse %r" % (a,))
        return from_sympy(parsed)
    if isinstance(a, SymPyBasic):
        from .interop import from_sympy
        return from_sympy(a)
    if isinstance(a, (tuple, list)):
        return Tuple(*[sympify(item) for item in a])
    if isinstance(a, Mapping):
        from .containers import Dict
        return Dict(*[(sympify(k), sympify(v)) for k, v in a.items()])
    raise SympifyError("cannot convert %r to an expression" % (a,))
