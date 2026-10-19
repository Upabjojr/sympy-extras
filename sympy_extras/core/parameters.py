"""Global parameters of the evaluation, with context managers.

``global_parameters.evaluate`` tells the evaluating functions (``add``,
``mul``, ``pow``, ``eq``, ..., ``mod``) whether to evaluate; the context
manager ``evaluate(False)`` turns them into the constructors, so that
``x + x`` is ``Add(x, x)`` inside the block, as in SymPy [1]_.
``distribute(False)`` keeps a number in front of a sum
(``2*(x + y)`` stays a product). ``global_parameters.divide_error`` makes
a division by zero raise instead of giving ``zoo`` (``seterr``).

References
==========

.. [1] SymPy Development Team, *SymPy documentation: parameters*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.parameters
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

__all__ = ['GlobalParameters', 'global_parameters', 'evaluate', 'distribute']


class GlobalParameters:
    """The switches read by the evaluating functions."""

    __slots__ = ('evaluate', 'distribute', 'divide_error')

    def __init__(self) -> None:
        self.evaluate: bool = True
        self.distribute: bool = True
        self.divide_error: bool = False


global_parameters = GlobalParameters()


@contextmanager
def evaluate(flag: bool) -> Iterator[None]:
    """A block in which the evaluating functions evaluate or not.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, evaluate
    >>> x = Symbol('x')
    >>> with evaluate(False):
    ...     e = x + x
    >>> e, e.args, x + x
    (x + x, (x, x), 2*x)
    """
    old = global_parameters.evaluate
    global_parameters.evaluate = flag
    try:
        yield
    finally:
        global_parameters.evaluate = old


@contextmanager
def distribute(flag: bool) -> Iterator[None]:
    """A block in which a number in front of a sum is distributed or not.

    Examples
    ========

    >>> from sympy_extras.core import symbols, distribute
    >>> x, y = symbols('x y')
    >>> with distribute(False):
    ...     e = 2*(x + y)
    >>> e, 2*(x + y)
    (2*(x + y), 2*x + 2*y)
    """
    old = global_parameters.distribute
    global_parameters.distribute = flag
    try:
        yield
    finally:
        global_parameters.distribute = old
