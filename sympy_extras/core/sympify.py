"""Conversion of Python and SymPy objects to the expression tree.

``sympify`` accepts the nodes of this core, Python integers, floats,
fractions and booleans, strings (parsed by SymPy's parser and converted),
tuples and lists (which become :class:`~sympy_extras.core.containers.Tuple`),
dictionaries (:class:`~sympy_extras.core.containers.Dict`) and SymPy objects
(converted by :func:`~sympy_extras.interop.from_sympy`).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence, Union

from .basic import Basic

__all__ = ['sympify', 'SympifyError', 'SympifyLike']

#: what ``sympify`` accepts (at run time a foreign object too, converted
#: by the registered bridge; typed callers convert it themselves)
SympifyLike = Union[Basic, int, float, Fraction, bool, str, Sequence['SympifyLike'],
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
    >>> sympify(float('inf')), sympify(float('-inf')), sympify(float('nan'))
    (oo, -oo, nan)
    >>> sympify('x**2 + 1')
    x**2 + 1
    >>> import sympy, sympy_extras.interop  # SymPy objects are converted by the bridge
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
        # an infinite float is the infinity of its sign and nan is nan, as in SymPy (the conversion to an exact
        # Float raised OverflowError and ValueError: latex(float('inf')) and x + float('inf') crashed)
        if a != a:
            from .numbers import nan
            return nan
        if a in (float('inf'), float('-inf')):
            from .numbers import oo
            return oo if a > 0 else -oo
        return Float(a)
    if isinstance(a, str):
        from .bridge import ForeignOperationUnavailable, bridge
        from .parameters import global_parameters
        from .parsing import ParseUnsupported, parse
        evaluate = global_parameters.evaluate
        try:
            if evaluate or not bridge().available():
                # the native parser; without evaluation SymPy's own (when
                # registered) keeps its literal forms (x - 1 is x + (-1)*1)
                return parse(a, evaluate)
        except ParseUnsupported as unsupported:
            reason = str(unsupported)
        else:
            reason = ''
        # outside the native grammar: the foreign system's parser, if any
        try:
            return bridge().parse(a, global_parameters.evaluate)
        except ForeignOperationUnavailable:
            # with what the native parser refused (it was "cannot parse" alone)
            raise SympifyError("cannot parse %r%s" % (a, ': ' + reason if reason else '')) from None
        except (SyntaxError, ValueError, TypeError, NameError) as error:
            raise SympifyError("cannot parse %r: %s" % (a, error)) from None
    if isinstance(a, (tuple, list)):
        return Tuple(*[sympify(item) for item in a])
    if isinstance(a, Mapping):
        from .containers import Dict
        return Dict(*[(sympify(k), sympify(v)) for k, v in a.items()])
    from .bridge import bridge
    if bridge().is_foreign(a):
        return bridge().from_foreign(a)
    raise SympifyError("cannot convert %r to an expression" % (a,))
